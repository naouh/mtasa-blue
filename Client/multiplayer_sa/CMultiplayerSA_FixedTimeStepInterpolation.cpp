/*****************************************************************************
 *
 *  PROJECT:     Multi Theft Auto
 *  LICENSE:     See LICENSE in the top level directory
 *  FILE:        multiplayer_sa/CMultiplayerSA_FixedTimeStepInterpolation.cpp
 *
 *  Multi Theft Auto is available from https://www.multitheftauto.com/
 *
 *****************************************************************************/
#include "StdInc.h"
#include <game/RenderWare.h>
#include "CMultiplayerSA_FixedTimeStep.h"
#include "../game_sa/CPoolSAInterface.h"
#include "../game_sa/CBikeSA.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

extern CCoreInterface* g_pCore;

// Render-side interpolation for the fixed time step mode.
//
// With the simulation stepping at a fixed rate, a frame rendered between two steps would show the
// world frozen at the last step. To render smoothly at any frame rate, every pooled entity (peds,
// vehicles, objects) and the camera keep the transform of the previous and the current step, and
// before each render the entity is placed at the interpolated transform. Before the next step the
// true transform is written back, so the simulation never sees interpolated values.
//
// Entities that something else moves between steps (scripts, MTA's own remote-entity interpolation,
// attached elements) are detected because their matrix no longer matches what we wrote; they are
// left alone and re-based on their new transform.
namespace
{
    constexpr std::uintptr_t ADDR_PedPool = 0xB74490;
    constexpr std::uintptr_t ADDR_VehiclePool = 0xB74494;
    constexpr std::uintptr_t ADDR_ObjectPool = 0xB7449C;
    constexpr std::uintptr_t ADDR_TheCamera = CLASS_CCamera;

    constexpr std::uintptr_t FUNC_CEntity_UpdateRwMatrix = 0x446F90;
    constexpr std::uintptr_t FUNC_CEntity_UpdateRwFrame = 0x532B00;
    constexpr std::uintptr_t FUNC_CPed_SetPedPositionInCar = 0x5DF910;
    constexpr std::uintptr_t FUNC_CCamera_CalculateDerivedValues = 0x5150E0;  // (bool bForMirror, bool bOriented)
    constexpr std::uintptr_t FUNC_RwFrameTransform = 0x7F0F70;                // RwFrame* RwFrameTransform(RwFrame*, const RwMatrix*, RwOpCombineType)

    // RwMatrix::flags for a matrix that is both normalised and orthogonal (rwMATRIXTYPEORTHONORMAL)
    constexpr std::uint32_t RW_MATRIX_ORTHONORMAL = 0x3;

    // Pool object strides, same values CPoolsSA uses to map an interface pointer back to its index
    constexpr std::size_t STRIDE_Ped = 1988;
    constexpr std::size_t STRIDE_Vehicle = 2584;
    constexpr std::size_t STRIDE_Object = 412;

    // Moves longer than this between two steps are teleports, not motion, and are not interpolated
    constexpr float TELEPORT_DISTANCE_SQ = 15.0f * 15.0f;

    struct Transform
    {
        CVector right, forward, up, pos;

        bool operator==(const Transform& o) const { return std::memcmp(this, &o, sizeof(Transform)) == 0; }
        bool operator!=(const Transform& o) const { return !(*this == o); }
        bool SameRotation(const Transform& o) const
        {
            return std::memcmp(&right, &o.right, sizeof(CVector) * 3) == 0;  // right, forward, up are contiguous
        }
        bool SamePosition(const Transform& o) const { return std::memcmp(&pos, &o.pos, sizeof(CVector)) == 0; }
    };

    Transform ReadTransform(const CMatrix_Padded& m)
    {
        return Transform{m.vRight, m.vFront, m.vUp, m.vPos};
    }

    void WriteTransform(CMatrix_Padded& m, const Transform& t)
    {
        m.vRight = t.right;
        m.vFront = t.forward;
        m.vUp = t.up;
        m.vPos = t.pos;
    }

    // The RenderWare camera frame uses the RW row order (right, up, at); the names in Transform are
    // only labels, the interpolation below never assumes which row points where.
    Transform ReadTransform(const RwMatrix& m)
    {
        return Transform{CVector(m.right.x, m.right.y, m.right.z), CVector(m.up.x, m.up.y, m.up.z), CVector(m.at.x, m.at.y, m.at.z),
                         CVector(m.pos.x, m.pos.y, m.pos.z)};
    }

    void WriteTransform(RwMatrix& m, const Transform& t)
    {
        m.right = {t.right.fX, t.right.fY, t.right.fZ};
        m.up = {t.forward.fX, t.forward.fY, t.forward.fZ};
        m.at = {t.up.fX, t.up.fY, t.up.fZ};
        m.pos = {t.pos.fX, t.pos.fY, t.pos.fZ};
    }

    CVector Lerp(const CVector& a, const CVector& b, float alpha)
    {
        return a + (b - a) * alpha;
    }

    float Dot(const CVector& a, const CVector& b)
    {
        return a.fX * b.fX + a.fY * b.fY + a.fZ * b.fZ;
    }

    bool Normalize(CVector& v)
    {
        const float len = v.Length();
        if (len < 1e-6f)
            return false;
        v *= 1.0f / len;
        return true;
    }

    // Linear interpolation of the basis followed by Gram-Schmidt re-orthonormalisation. Rotation between
    // two consecutive steps is small (a few degrees at most), so this is indistinguishable from a slerp.
    // Gram-Schmidt keeps the handedness of the input, so it works for entity matrices and for the
    // RenderWare camera frame alike without knowing which row is which.
    bool Interpolate(const Transform& a, const Transform& b, float alpha, Transform& out)
    {
        out.pos = Lerp(a.pos, b.pos, alpha);
        out.up = Lerp(a.up, b.up, alpha);
        out.forward = Lerp(a.forward, b.forward, alpha);
        out.right = Lerp(a.right, b.right, alpha);

        if (!Normalize(out.up))
            return false;
        out.forward -= out.up * Dot(out.forward, out.up);
        if (!Normalize(out.forward))
            return false;
        out.right -= out.up * Dot(out.right, out.up);
        out.right -= out.forward * Dot(out.right, out.forward);
        return Normalize(out.right);
    }

    struct TrackedSlot
    {
        Transform    prev{};
        Transform    cur{};
        Transform    written{};   // what Apply last wrote, to detect outside modifications
        std::uint8_t poolId = 0;  // tPoolObjectFlags::nId, changes when the slot is reused
        bool         valid = false;
        bool         applied = false;
        bool         inVehicle = false;  // peds only: seated in a vehicle, placed from the vehicle instead of interpolated
        Transform    shownAtStep{};      // extrapolation: where it was displayed when the last step arrived
        bool         hasShownAtStep = false;
        bool         wasShown = false;  // Apply wrote this slot since the last capture (Restore clears `applied`, not this)
        unsigned     shownFrame = 0;    // frame id of the last Apply that wrote this slot (Reapply uses it)
        // A script placed the entity during the render phase (onClientPreRender / onClientRender). Scripts that do
        // this do it every frame (rewind, attachments, podium) and expect to see exactly what they wrote: the next
        // frame shows `scriptValue` as is, no prediction, instead of the simulation's step-quantised motion on top.
        bool         scriptDriven = false;  // camera: the whole pose
        bool         posScripted = false;   // entities: per component, a rotation-only write (podium) must keep the
        bool         rotScripted = false;   // position's prediction or the car moves in step-sized jolts
        Transform    scriptValue{};
    };

    struct PoolTracker
    {
        std::uintptr_t           poolPtrAddr;
        std::size_t              stride;
        std::vector<TrackedSlot> slots;

        CPoolSAInterface<CEntitySAInterface>* Pool() const { return *reinterpret_cast<CPoolSAInterface<CEntitySAInterface>**>(poolPtrAddr); }

        // Entity of a live slot, or nullptr. `expectedId` additionally requires the slot not to have been reused.
        CEntitySAInterface* Entity(const CPoolSAInterface<CEntitySAInterface>* pool, int i, const std::uint8_t* expectedId = nullptr) const
        {
            const tPoolObjectFlags flags = pool->m_byteMap[i];
            if (flags.bEmpty)
                return nullptr;
            if (expectedId && flags.nId != *expectedId)
                return nullptr;
            auto* entity = reinterpret_cast<CEntitySAInterface*>(reinterpret_cast<std::uint8_t*>(pool->m_pObjects) + i * stride);
            return entity->matrix ? entity : nullptr;
        }
    };

    PoolTracker s_pools[] = {
        {ADDR_PedPool, STRIDE_Ped, {}},
        {ADDR_VehiclePool, STRIDE_Vehicle, {}},
        {ADDR_ObjectPool, STRIDE_Object, {}},
    };

    TrackedSlot s_camera;
    bool        s_anythingApplied = false;
    unsigned    s_frameId = 0;  // incremented by each Apply (once per rendered frame)

    // CPad::NewMouseControllerState.m_AmountMoved (0xB73418 + 0xC): the mouse delta the last simulation step consumed.
    // CPad::UpdatePads refreshes it once per CGame::Process, so it tells whether the step's camera motion was mouse driven.
    constexpr std::uintptr_t ADDR_CPad_NewMouseControllerState_AmountMoved = 0xB73424;
    bool                     s_mouseMovedLastStep = false;
    unsigned    s_rebased[4] = {0, 0, 0, 0};  // TEMP: entities re-based per pool (peds, vehicles, objects) + camera since the last dump

    // The camera is interpolated on the RenderWare camera frame itself, which is what the renderer
    // actually uses. CCamera's own matrix and derived values (frustum planes) are left as the last
    // simulation step produced them; they are at most one step of motion away.
    RwFrame* CameraFrame()
    {
        auto* camera = reinterpret_cast<CCameraSAInterface*>(ADDR_TheCamera);
        if (!camera->m_pRwCamera)
            return nullptr;
        return reinterpret_cast<RwFrame*>(camera->m_pRwCamera->object.object.parent);
    }

    bool ReadCameraTransform(Transform& out)
    {
        RwFrame* frame = CameraFrame();
        if (!frame)
            return false;
        out = ReadTransform(frame->modelling);
        return true;
    }

    // Rigid attachment helpers: express `world` in the frame of `e` and back. Rows of a Transform are the axes.
    CVector ToLocalPos(const Transform& e, const CVector& world)
    {
        const CVector d = world - e.pos;
        return CVector(Dot(d, e.right), Dot(d, e.forward), Dot(d, e.up));
    }
    CVector ToLocalDir(const Transform& e, const CVector& d)
    {
        return CVector(Dot(d, e.right), Dot(d, e.forward), Dot(d, e.up));
    }
    CVector ToWorldDir(const Transform& e, const CVector& l)
    {
        return e.right * l.fX + e.forward * l.fY + e.up * l.fZ;
    }

    Transform ToLocal(const Transform& e, const Transform& world)
    {
        return Transform{ToLocalDir(e, world.right), ToLocalDir(e, world.forward), ToLocalDir(e, world.up), ToLocalPos(e, world.pos)};
    }
    Transform ToWorld(const Transform& e, const Transform& local)
    {
        return Transform{ToWorldDir(e, local.right), ToWorldDir(e, local.forward), ToWorldDir(e, local.up), e.pos + ToWorldDir(e, local.pos)};
    }

    // The camera carried over to where its target is displayed. Its pose relative to the target is what the
    // simulation computed, interpolated between the last two steps so mouse orbit stays smooth (one step late)
    // while the target itself never moves in frame.
    Transform AttachCamera(const TrackedSlot& target, const TrackedSlot& camera, float blend)
    {
        const Transform localCur = ToLocal(target.cur, camera.cur);
        Transform       local = localCur;
        if (blend < 1.0f)
        {
            const Transform localPrev = ToLocal(target.prev, camera.prev);
            if (!Interpolate(localPrev, localCur, blend, local))
                local = localCur;
        }
        return ToWorld(target.written, local);
    }

    // Anchored camera (default while extrapolating): the camera keeps the simulation's own pose, interpolated
    // between the last two steps (smooth mouse orbit, one step late), but its position is measured relative to
    // its target and re-anchored to where the target is displayed. The target never jitters in frame, and the
    // target's rotation (bumps, rolls) is not passed on to the camera, unlike a rigid attachment.
    Transform AnchorCamera(const TrackedSlot& target, const TrackedSlot& camera, float blend)
    {
        const CVector targetShown = target.applied ? target.written.pos : target.cur.pos;
        const CVector offsetPrev = camera.prev.pos - target.prev.pos;
        const CVector offsetCur = camera.cur.pos - target.cur.pos;
        // A cut (respawn, spectate switch, teleport) between the two steps: no blending through the gap
        if ((offsetCur - offsetPrev).LengthSquared() > TELEPORT_DISTANCE_SQ || (camera.cur.pos - camera.prev.pos).LengthSquared() > TELEPORT_DISTANCE_SQ)
            blend = 1.0f;
        Transform out;
        if (blend >= 1.0f || !Interpolate(camera.prev, camera.cur, blend, out))
            out = camera.cur;
        out.pos = targetShown + Lerp(offsetPrev, offsetCur, blend);
        return out;
    }

    // Tracked slot of what the active camera follows (a seated ped counts as its vehicle), when it was placed this
    // frame. Only follow modes qualify: a fixed or scripted camera (setCameraMatrix, cutscenes, fly-bys) does not move
    // with its target, so carrying it along with the target's extrapolation would displace it by one step of motion.
    const TrackedSlot* FindCameraTargetSlot()
    {
        auto* camera = reinterpret_cast<CCameraSAInterface*>(ADDR_TheCamera);
        if (camera->m_bLookingAtVector || camera->ActiveCam >= 3)
            return nullptr;
        const CCamSAInterface& cam = camera->Cams[camera->ActiveCam];
        if (cam.Mode == MODE_NONE || cam.Mode == MODE_FIXED || cam.Mode == MODE_FLYBY)
            return nullptr;
        CEntitySAInterface* target = cam.CamTargetEntity;
        if (!target)
            return nullptr;
        if (target->nType == 3 /* ENTITY_TYPE_PED */)
        {
            auto* ped = reinterpret_cast<CPedSAInterface*>(target);
            if (ped->pedFlags.bInVehicle && ped->pVehicle)
                target = reinterpret_cast<CEntitySAInterface*>(ped->pVehicle);
        }
        for (PoolTracker& tracker : s_pools)
        {
            auto* pool = tracker.Pool();
            if (!pool || !pool->m_pObjects)
                continue;
            const std::ptrdiff_t off = reinterpret_cast<std::uint8_t*>(target) - reinterpret_cast<std::uint8_t*>(pool->m_pObjects);
            if (off < 0 || off % static_cast<std::ptrdiff_t>(tracker.stride) != 0)
                continue;
            const std::size_t i = static_cast<std::size_t>(off / static_cast<std::ptrdiff_t>(tracker.stride));
            if (i >= tracker.slots.size())
                continue;
            const TrackedSlot& s = tracker.slots[i];
            return (s.valid && !s.inVehicle) ? &s : nullptr;
        }
        return nullptr;
    }

    void WriteCameraTransform(const Transform& t)
    {
        RwFrame* frame = CameraFrame();
        if (!frame)
            return;
        RwMatrix m = frame->modelling;
        WriteTransform(m, t);
        m.flags = RW_MATRIX_ORTHONORMAL;
        // Replaces the frame's modelling matrix and marks the hierarchy dirty, so the camera's view matrix
        // is rebuilt from it on the next RwCameraBeginUpdate
        reinterpret_cast<RwFrame*(__cdecl*)(RwFrame*, const RwMatrix*, RwOpCombineType)>(FUNC_RwFrameTransform)(frame, &m, rwCOMBINEREPLACE);

        // Everything that bills itself against the camera without going through RenderWare (coronas, 3D
        // markers, particles, MTA's own world-to-screen) reads CCamera's matrix, so keep it in step. The game
        // copies that matrix into the RW frame as right -> right, forward -> at, up -> up (CCamera::CopyCameraMatrixToRWCam).
        CMatrix_Padded& cm = reinterpret_cast<CCameraSAInterface*>(ADDR_TheCamera)->m_cameraMatrix;
        cm.vRight = t.right;
        cm.vFront = t.up;    // RW "at"
        cm.vUp = t.forward;  // RW "up"
        cm.vPos = t.pos;

        if (FixedTimeStep::GetDebugFlags() & FixedTimeStep::DEBUG_CAMERA_DERIVED_VALUES)
            reinterpret_cast<void(__thiscall*)(CCameraSAInterface*, bool, bool)>(FUNC_CCamera_CalculateDerivedValues)(
                reinterpret_cast<CCameraSAInterface*>(ADDR_TheCamera), false, false);
    }

    // True when `entity` is a live object of the vehicle pool (a stale pVehicle pointer would otherwise be followed)
    bool IsPooledVehicle(const void* entity)
    {
        const PoolTracker& tracker = s_pools[1];
        auto*              pool = tracker.Pool();
        if (!entity || !pool || !pool->m_pObjects || !pool->m_byteMap)
            return false;
        const std::ptrdiff_t off = reinterpret_cast<const std::uint8_t*>(entity) - reinterpret_cast<const std::uint8_t*>(pool->m_pObjects);
        if (off < 0 || off % static_cast<std::ptrdiff_t>(tracker.stride) != 0)
            return false;
        const std::ptrdiff_t i = off / static_cast<std::ptrdiff_t>(tracker.stride);
        return i < pool->m_nSize && !pool->m_byteMap[i].bEmpty;
    }

    bool IsPedSeatedInVehicle(const CPedSAInterface* ped)
    {
        return ped->pedFlags.bInVehicle && IsPooledVehicle(ped->pVehicle);
    }

    void ResetPedPreRenderFlags()
    {
        PoolTracker& tracker = s_pools[0];
        auto*        pool = tracker.Pool();
        if (!pool || !pool->m_pObjects || !pool->m_byteMap)
            return;

        for (int i = 0; i < pool->m_nSize; ++i)
        {
            if (pool->m_byteMap[i].bEmpty)
                continue;
            auto* ped = reinterpret_cast<CPedSAInterface*>(reinterpret_cast<std::uint8_t*>(pool->m_pObjects) + i * tracker.stride);
            ped->pedFlags.bCalledPreRender = false;
            // CPed::PreRenderAfterTest raises this on the player model after rotating the shoulders, and
            // CEntity::UpdateAnim (simulation step) is what lowers it again. While it is raised
            // CEntity::UpdateRpHAnim does not rebuild the skeleton, so on a rendered frame without a step
            // the bones would stay where the previous frame put them while the root moved on.
            ped->bDontUpdateHierarchy = false;
        }
    }

    // CPed::SetPedPositionInCar: the game's own placement of a seated ped from its vehicle's matrix (what the
    // simulation runs every step). Re-run here after the vehicle moved so the ped stays in its seat.
    void PlaceSeatedPeds(bool markApplied)
    {
        PoolTracker& tracker = s_pools[0];
        auto*        pool = tracker.Pool();
        if (!pool || !pool->m_pObjects || !pool->m_byteMap || tracker.slots.size() != static_cast<std::size_t>(pool->m_nSize))
            return;

        for (int i = 0; i < pool->m_nSize; ++i)
        {
            TrackedSlot& slot = tracker.slots[i];
            if (!slot.valid || !slot.inVehicle)
                continue;
            if (!markApplied && !slot.applied)
                continue;

            auto* entity = tracker.Entity(pool, i, &slot.poolId);
            if (!entity || !IsPedSeatedInVehicle(reinterpret_cast<CPedSAInterface*>(entity)))
            {
                slot.valid = false;
                slot.applied = false;
                continue;
            }

            // On a bike the rider is placed from CBike::m_mLeanMatrix, which CBike::CalculateLeanMatrix builds from the
            // bike's matrix once per simulation step and then caches (m_bLeanMatrixCalculated). The bike has just been
            // moved, so drop the cache or the rider stays seated on the previous step's bike and shakes in the saddle.
            auto* vehicle = reinterpret_cast<CVehicleSAInterface*>(reinterpret_cast<CPedSAInterface*>(entity)->pVehicle);
            if (vehicle->m_vehicleClass == VehicleClass::BIKE || vehicle->m_vehicleClass == VehicleClass::BMX)
                reinterpret_cast<CBikeSAInterface*>(vehicle)->m_bLeanMatrixCalculated = 0;

            reinterpret_cast<void(__thiscall*)(CEntitySAInterface*)>(FUNC_CPed_SetPedPositionInCar)(entity);
            reinterpret_cast<void(__thiscall*)(CEntitySAInterface*)>(FUNC_CEntity_UpdateRwMatrix)(entity);
            reinterpret_cast<void(__thiscall*)(CEntitySAInterface*)>(FUNC_CEntity_UpdateRwFrame)(entity);
            slot.applied = markApplied;
            if (markApplied)
                s_anythingApplied = true;

            // TEMP (development only): numbers behind the "driver trembles" report, once per second for the first seated ped
            if (markApplied && (FixedTimeStep::GetDebugFlags() & FixedTimeStep::DEBUG_DUMP_CAMERA))
            {
                static unsigned long lastDump = 0;
                const unsigned long  now = GetTickCount32();
                if (now - lastDump >= 1000)
                {
                    lastDump = now;
                    auto*          ped = reinterpret_cast<CPedSAInterface*>(entity);
                    const CVector& pedPos = entity->matrix->vPos;
                    const CVector& vehPos = ped->pVehicle->matrix ? ped->pVehicle->matrix->vPos : ped->pVehicle->m_transform.m_translate;
                    CVector        ltmPos = pedPos;
                    if (entity->m_pRwObject)
                    {
                        RwMatrix* ltm = reinterpret_cast<RwMatrix*(__cdecl*)(void*)>(0x7F0990)(entity->m_pRwObject->object.parent);  // RwFrameGetLTM
                        ltmPos = CVector(ltm->pos.x, ltm->pos.y, ltm->pos.z);
                    }
                    g_pCore->GetConsole()->Printf("[fts] seated ped %d: matrix-veh %.3f m | ltm-matrix %.3f m | flags inVeh=%d calledPreRender=%d", i,
                                                  (pedPos - vehPos).Length(), (ltmPos - pedPos).Length(), ped->pedFlags.bInVehicle ? 1 : 0,
                                                  ped->pedFlags.bCalledPreRender ? 1 : 0);
                }
            }
        }
    }

    void WriteEntity(CEntitySAInterface* entity, const Transform& t)
    {
        WriteTransform(*entity->matrix, t);
        entity->m_transform.m_translate = t.pos;
        // Same two calls CEntitySA::SetMatrix makes: CEntity::UpdateRwMatrix copies the matrix into the clump's root
        // frame, CEntity::UpdateRwFrame marks the frame hierarchy dirty so every child atomic (vehicle body, doors,
        // wheels) is re-placed from it. Without the second one the wheels and the body end up with different transforms.
        reinterpret_cast<void(__thiscall*)(CEntitySAInterface*)>(FUNC_CEntity_UpdateRwMatrix)(entity);
        reinterpret_cast<void(__thiscall*)(CEntitySAInterface*)>(FUNC_CEntity_UpdateRwFrame)(entity);
    }

    void CaptureSlot(TrackedSlot& slot, const Transform& now, std::uint8_t poolId)
    {
        // Extrapolation shows a prediction; when the real step arrives it rarely matches exactly. Remember
        // what was on screen so the display can slide from there to the new prediction instead of snapping.
        slot.hasShownAtStep = slot.valid && slot.poolId == poolId && slot.wasShown;
        if (slot.hasShownAtStep)
            slot.shownAtStep = slot.written;
        slot.wasShown = false;
        if (slot.valid && slot.poolId == poolId)
            slot.prev = slot.cur;
        else
            slot.prev = now;
        slot.cur = now;
        slot.poolId = poolId;
        slot.valid = true;
        slot.applied = false;
    }

    // `x` moved along with a change that took the entity from `before` to `after`
    Transform Carry(const Transform& x, const Transform& before, const Transform& after)
    {
        Transform out = x;
        out.pos = x.pos + (after.pos - before.pos);
        if (after.SameRotation(before))
            return out;
        if (x.SameRotation(before))
        {
            out.right = after.right;
            out.forward = after.forward;
            out.up = after.up;
            return out;
        }
        out.right = ToWorldDir(after, ToLocalDir(before, x.right));
        out.forward = ToWorldDir(after, ToLocalDir(before, x.forward));
        out.up = ToWorldDir(after, ToLocalDir(before, x.up));
        return out;
    }

    // Something other than the simulation (MTA's sync correction, a script, an attachment) took the entity from
    // `before` to `now`. Such writes are relative to what they read: remote vehicles get `position + correction` every
    // frame, setElementRotation keeps the position it finds. So the change is applied as a delta to the whole history,
    // never adopted as an absolute state: when `before` is a displayed transform that would turn the extrapolation into
    // simulation state and it would compound on every write. Shifting `prev` along also keeps the correction out of the
    // velocity the extrapolation projects. A jump is a teleport and restarts the history.
    void AdoptExternalChange(TrackedSlot& slot, const Transform& before, const Transform& now)
    {
        if ((now.pos - before.pos).LengthSquared() > TELEPORT_DISTANCE_SQ)
        {
            slot.prev = now;
            slot.cur = now;
            slot.hasShownAtStep = false;
            return;
        }
        slot.prev = Carry(slot.prev, before, now);
        slot.cur = Carry(slot.cur, before, now);
        if (slot.hasShownAtStep)
            slot.shownAtStep = Carry(slot.shownAtStep, before, now);
        if (slot.scriptDriven || slot.posScripted || slot.rotScripted)
            slot.scriptValue = Carry(slot.scriptValue, before, now);
    }

    // A script wrote the entity while it was displayed (render phase). What it wrote is absolute, but only the part
    // it changed: setElementRotation keeps the displayed position it finds, which must not become the simulation's.
    // The history collapses onto that value: the script drives the entity frame by frame, there is no step motion to
    // predict, and `written` becomes what was actually rendered so Reapply/Restore recognise it.
    void AdoptRenderPhaseWrite(TrackedSlot& slot, const Transform& now)
    {
        Transform truth = slot.cur;
        if (!now.SamePosition(slot.written))
        {
            truth.pos = now.pos;
            slot.prev.pos = now.pos;  // no step motion to predict on this component, the script moves it
            slot.posScripted = true;
            slot.hasShownAtStep = false;
        }
        if (!now.SameRotation(slot.written))
        {
            truth.right = now.right;
            truth.forward = now.forward;
            truth.up = now.up;
            slot.prev.right = now.right;
            slot.prev.forward = now.forward;
            slot.prev.up = now.up;
            slot.rotScripted = true;
        }
        slot.cur = truth;
        slot.scriptValue = truth;
        slot.written = now;
    }

    bool ComputeInterpolated(const TrackedSlot& slot, float alpha, Transform& out)
    {
        if ((slot.cur.pos - slot.prev.pos).LengthSquared() > TELEPORT_DISTANCE_SQ)
            return false;

        const bool extrapolating = alpha > 1.0f;
        if (slot.prev == slot.cur && !(extrapolating && slot.hasShownAtStep))
            return false;

        Transform target;
        if (slot.prev == slot.cur)
            target = slot.cur;
        else if (!Interpolate(slot.prev, slot.cur, alpha, target))
            return false;

        // Correction smoothing: over the step that follows a new simulation state, blend from where the entity
        // was displayed towards the new prediction (weight 0 right after the step, 1 just before the next one)
        if (extrapolating && slot.hasShownAtStep)
        {
            const float w = std::min(alpha - 1.0f, 1.0f);
            if (Interpolate(slot.shownAtStep, target, w, out))
                return true;
        }
        out = target;
        return true;
    }
}  // namespace

void FixedTimeStep::Interpolation::Capture()
{
    for (PoolTracker& tracker : s_pools)
    {
        auto* pool = tracker.Pool();
        if (!pool || !pool->m_pObjects || !pool->m_byteMap || pool->m_nSize <= 0)
        {
            tracker.slots.clear();
            continue;
        }

        if (tracker.slots.size() != static_cast<std::size_t>(pool->m_nSize))
            tracker.slots.assign(pool->m_nSize, TrackedSlot{});

        for (int i = 0; i < pool->m_nSize; ++i)
        {
            TrackedSlot& slot = tracker.slots[i];
            auto*        entity = tracker.Entity(pool, i);
            if (!entity)
            {
                slot.valid = false;
                slot.applied = false;
                continue;
            }
            CaptureSlot(slot, ReadTransform(*entity->matrix), pool->m_byteMap[i].nId);
            slot.inVehicle = (tracker.poolPtrAddr == ADDR_PedPool) && !(FixedTimeStep::GetDebugFlags() & FixedTimeStep::DEBUG_LERP_SEATED_PEDS) &&
                             IsPedSeatedInVehicle(reinterpret_cast<CPedSAInterface*>(entity));
        }
    }

    Transform cameraNow;
    if (ReadCameraTransform(cameraNow))
        CaptureSlot(s_camera, cameraNow, 0);
    else
        s_camera.valid = false;

    const float* mouseMoved = reinterpret_cast<const float*>(ADDR_CPad_NewMouseControllerState_AmountMoved);
    s_mouseMovedLastStep = mouseMoved[0] != 0.0f || mouseMoved[1] != 0.0f;

    // TEMP (development only): establish how CCamera's matrix rows map onto the RW camera frame rows
    if ((FixedTimeStep::GetDebugFlags() & DEBUG_DUMP_CAMERA) && s_camera.valid)
    {
        static unsigned long lastDump = 0;
        const unsigned long  now = GetTickCount32();
        if (now - lastDump >= 1000)
        {
            lastDump = now;
            const CMatrix_Padded& cm = reinterpret_cast<CCameraSAInterface*>(ADDR_TheCamera)->m_cameraMatrix;
            g_pCore->GetConsole()->Printf("[fts] CCamera  right %.3f %.3f %.3f | front %.3f %.3f %.3f | up %.3f %.3f %.3f | pos %.1f %.1f %.1f", cm.vRight.fX,
                                          cm.vRight.fY, cm.vRight.fZ, cm.vFront.fX, cm.vFront.fY, cm.vFront.fZ, cm.vUp.fX, cm.vUp.fY, cm.vUp.fZ, cm.vPos.fX,
                                          cm.vPos.fY, cm.vPos.fZ);
            const Transform& rw = s_camera.cur;
            g_pCore->GetConsole()->Printf("[fts] RwFrame  right %.3f %.3f %.3f | up    %.3f %.3f %.3f | at %.3f %.3f %.3f | pos %.1f %.1f %.1f", rw.right.fX,
                                          rw.right.fY, rw.right.fZ, rw.forward.fX, rw.forward.fY, rw.forward.fZ, rw.up.fX, rw.up.fY, rw.up.fZ, rw.pos.fX,
                                          rw.pos.fY, rw.pos.fZ);
            g_pCore->GetConsole()->Printf(
                "[fts] re-based since last dump: peds %u | vehicles %u | objects %u | camera %u (each one is a frame rendered without interpolation)",
                s_rebased[0], s_rebased[1], s_rebased[2], s_rebased[3]);
            s_rebased[0] = s_rebased[1] = s_rebased[2] = s_rebased[3] = 0;
        }
    }
}

void FixedTimeStep::Interpolation::Apply(float alpha, float cameraPosAlpha, float cameraRotAlpha)
{
    Transform          t;
    const unsigned int debugFlags = FixedTimeStep::GetDebugFlags();
    ++s_frameId;

    for (PoolTracker& tracker : s_pools)
    {
        if (debugFlags & DEBUG_NO_ENTITIES)
            break;
        if ((debugFlags & DEBUG_VEHICLES_ONLY) && tracker.poolPtrAddr != ADDR_VehiclePool)
            continue;

        auto* pool = tracker.Pool();
        if (!pool || !pool->m_pObjects || !pool->m_byteMap || tracker.slots.size() != static_cast<std::size_t>(pool->m_nSize))
            continue;

        for (int i = 0; i < pool->m_nSize; ++i)
        {
            TrackedSlot& slot = tracker.slots[i];
            if (!slot.valid || slot.inVehicle)
                continue;

            // The entity may have been destroyed or replaced since the capture (scripts run between steps)
            auto* entity = tracker.Entity(pool, i, &slot.poolId);
            if (!entity)
            {
                slot.valid = false;
                continue;
            }

            // MTA's pulses ran on the true world between the last step and this call (sync corrections land here)
            const Transform now = ReadTransform(*entity->matrix);
            if (now != slot.cur)
                AdoptExternalChange(slot, slot.cur, now);

            bool show = ComputeInterpolated(slot, alpha, t);
            if (!show && (slot.posScripted || slot.rotScripted))
            {
                t = slot.cur;
                show = true;
            }
            if (!show)
                continue;

            // What a script wrote last frame is shown as is on that component, the rest keeps its prediction
            if (slot.posScripted)
                t.pos = slot.scriptValue.pos;
            if (slot.rotScripted)
            {
                t.right = slot.scriptValue.right;
                t.forward = slot.scriptValue.forward;
                t.up = slot.scriptValue.up;
            }
            slot.posScripted = false;
            slot.rotScripted = false;

            WriteEntity(entity, t);
            slot.written = t;
            slot.applied = true;
            slot.wasShown = true;
            slot.shownFrame = s_frameId;
            s_anythingApplied = true;
        }
    }

    // Seated peds follow their (now interpolated) vehicle through the game's own seat placement, exactly as
    // the simulation does each step. Done after the pools loop because vehicles come after peds in it.
    if (!(debugFlags & (DEBUG_NO_ENTITIES | DEBUG_VEHICLES_ONLY)))
        PlaceSeatedPeds(true);

    // CPed::PreRenderAfterTest only rebuilds the skeleton (UpdateRpHAnim) when bCalledPreRender is clear,
    // and only CPed::ProcessControl clears it. On a rendered frame without a simulation step the bones
    // would keep last frame's world matrices while the root moved: clear it ourselves, once per frame.
    ResetPedPreRenderFlags();

    // Camera. Default while extrapolating: anchored to the displayed target (AnchorCamera), so the target never
    // jitters in frame and the mouse orbit, which no projection predicts, stays the simulation's own, one step
    // late. 256 extrapolates the camera like an entity instead, 1024 tries the rigid attachment. Without
    // extrapolation the camera is interpolated like everything else.
    Transform          cameraNow, camPos, camRot;
    bool               camOk = false;
    // With the camera extrapolated (256) the mouse is the one thing no projection predicts: a flick that starts or
    // stops between two steps overshoots and corrects, a visible jolt. Steps that consumed a mouse delta therefore
    // fall back to the anchored pose (one step late on the mouse, which the eye does not notice) and the steps
    // driven by the vehicle alone keep the prediction (no steering latency).
    const bool         anchorMode = alpha > 1.0f && (!(debugFlags & DEBUG_EXTRAPOLATE_CAMERA) || s_mouseMovedLastStep);
    const TrackedSlot* anchorTo = anchorMode ? FindCameraTargetSlot() : nullptr;
    if (!(debugFlags & DEBUG_NO_CAMERA) && s_camera.valid && s_camera.scriptDriven)
    {
        // setCameraMatrix during the render phase (cinematics, podium sweep, rewind): shown exactly as written
        t = s_camera.scriptValue;
        s_camera.scriptDriven = false;
        camOk = true;
    }
    else if (!(debugFlags & DEBUG_NO_CAMERA) && s_camera.valid && anchorTo)
    {
        const float blend = std::min(alpha - 1.0f, 1.0f);
        t = (debugFlags & DEBUG_ATTACH_CAMERA) ? AttachCamera(*anchorTo, s_camera, blend) : AnchorCamera(*anchorTo, s_camera, blend);
        // Same correction smoothing as the entities, so switching between the predicted and the anchored pose from
        // one step to the next slides over the step instead of snapping. The position is smoothed as an offset from
        // the target, never as an absolute position: smoothing the absolute pose makes the camera trail the target
        // it is anchored to, and the vehicle wobbles in frame.
        if (s_camera.hasShownAtStep && anchorTo->hasShownAtStep && !(debugFlags & DEBUG_ATTACH_CAMERA))
        {
            const CVector targetShown = anchorTo->applied ? anchorTo->written.pos : anchorTo->cur.pos;
            const CVector offsetShown = s_camera.shownAtStep.pos - anchorTo->shownAtStep.pos;
            const CVector offsetNow = t.pos - targetShown;
            Transform     smoothed;
            if ((offsetNow - offsetShown).LengthSquared() <= TELEPORT_DISTANCE_SQ && Interpolate(s_camera.shownAtStep, t, blend, smoothed))
            {
                smoothed.pos = targetShown + Lerp(offsetShown, offsetNow, blend);
                t = smoothed;
            }
        }
        camOk = true;
    }
    else if (!(debugFlags & DEBUG_NO_CAMERA) && s_camera.valid)
    {
        const bool okPos = ComputeInterpolated(s_camera, cameraPosAlpha, camPos);
        const bool okRot = ComputeInterpolated(s_camera, cameraRotAlpha, camRot);
        camOk = okPos || okRot;
        if (camOk)
        {
            t = okRot ? camRot : camPos;
            if (okPos)
                t.pos = camPos.pos;
        }
    }
    if (camOk && ReadCameraTransform(cameraNow))
    {
        if (cameraNow != s_camera.cur)
        {
            // Moved by MTA or a script since the last step (a fixed camera re-applied each pulse): nothing to predict
            // from that, it is where it is
            s_camera.prev = cameraNow;
            s_camera.cur = cameraNow;
            s_camera.hasShownAtStep = false;
        }
        else
        {
            WriteCameraTransform(t);
            s_camera.written = t;
            s_camera.applied = true;
            s_camera.wasShown = true;
            s_camera.shownFrame = s_frameId;
            s_anythingApplied = true;
        }
    }
}

// Second placement of the same frame, before onClientRender: the network and sync code ran on the restored world in
// between, so the entities are put back exactly where this frame was drawn (what was rendered is what the scripts
// project to the screen). Changes made meanwhile are absorbed as deltas, like in Apply, so Restore keeps them.
void FixedTimeStep::Interpolation::Reapply()
{
    for (PoolTracker& tracker : s_pools)
    {
        auto* pool = tracker.Pool();
        if (!pool || !pool->m_pObjects || !pool->m_byteMap || tracker.slots.size() != static_cast<std::size_t>(pool->m_nSize))
            continue;

        for (int i = 0; i < pool->m_nSize; ++i)
        {
            TrackedSlot& slot = tracker.slots[i];
            if (!slot.valid || slot.inVehicle || slot.shownFrame != s_frameId || slot.applied)
                continue;
            auto* entity = tracker.Entity(pool, i, &slot.poolId);
            if (!entity)
            {
                slot.valid = false;
                continue;
            }
            const Transform now = ReadTransform(*entity->matrix);
            if (now != slot.cur)
            {
                AdoptExternalChange(slot, slot.cur, now);
                if (slot.prev == slot.cur)
                    continue;  // teleported since the frame was drawn (map change, respawn): stays where it was put
            }
            WriteEntity(entity, slot.written);
            slot.applied = true;
            s_anythingApplied = true;
        }
    }

    if (!(FixedTimeStep::GetDebugFlags() & (DEBUG_NO_ENTITIES | DEBUG_VEHICLES_ONLY)))
        PlaceSeatedPeds(true);

    Transform cameraNow;
    if (s_camera.valid && s_camera.shownFrame == s_frameId && !s_camera.applied && ReadCameraTransform(cameraNow))
    {
        if (cameraNow != s_camera.cur)
        {
            // A script set the camera from a timer or event after the frame: it stays, as written
            s_camera.prev = cameraNow;
            s_camera.cur = cameraNow;
            s_camera.hasShownAtStep = false;
        }
        else
        {
            WriteCameraTransform(s_camera.written);
            s_camera.applied = true;
            s_anythingApplied = true;
        }
    }
}

void FixedTimeStep::Interpolation::Restore()
{
    if (!s_anythingApplied)
        return;
    s_anythingApplied = false;

    for (PoolTracker& tracker : s_pools)
    {
        auto* pool = tracker.Pool();
        if (!pool || !pool->m_pObjects || !pool->m_byteMap || tracker.slots.size() != static_cast<std::size_t>(pool->m_nSize))
            continue;

        for (int i = 0; i < pool->m_nSize; ++i)
        {
            TrackedSlot& slot = tracker.slots[i];
            if (!slot.applied || slot.inVehicle)
                continue;
            slot.applied = false;

            auto* entity = tracker.Entity(pool, i, &slot.poolId);
            if (!entity)
            {
                slot.valid = false;
                continue;
            }

            const Transform now = ReadTransform(*entity->matrix);
            if (now != slot.written)
            {
                AdoptRenderPhaseWrite(slot, now);  // a script moved it while displayed (onClientPreRender/Render)
                s_rebased[&tracker - s_pools]++;
            }
            WriteEntity(entity, slot.cur);
        }
    }

    // Vehicles are back at their true transform: seat their peds there again
    PlaceSeatedPeds(false);

    if (s_camera.applied)
    {
        s_camera.applied = false;
        Transform now;
        if (!ReadCameraTransform(now))
            s_camera.valid = false;
        else if (now == s_camera.written)
            WriteCameraTransform(s_camera.cur);
        else
        {
            // setCameraMatrix during the render phase: absolute, shown as is next frame, nothing to predict from it
            s_camera.prev = now;
            s_camera.cur = now;
            s_camera.hasShownAtStep = false;
            s_camera.scriptDriven = true;
            s_camera.scriptValue = now;
            s_camera.written = now;
        }
        s_rebased[3]++;
    }
}

void FixedTimeStep::Interpolation::AbsorbExternalChanges()
{
    for (PoolTracker& tracker : s_pools)
    {
        auto* pool = tracker.Pool();
        if (!pool || !pool->m_pObjects || !pool->m_byteMap || tracker.slots.size() != static_cast<std::size_t>(pool->m_nSize))
            continue;

        for (int i = 0; i < pool->m_nSize; ++i)
        {
            TrackedSlot& slot = tracker.slots[i];
            if (!slot.valid || slot.inVehicle || slot.applied)
                continue;
            auto* entity = tracker.Entity(pool, i, &slot.poolId);
            if (!entity)
            {
                slot.valid = false;
                continue;
            }
            const Transform now = ReadTransform(*entity->matrix);
            if (now != slot.cur)
                AdoptExternalChange(slot, slot.cur, now);
        }
    }
}

void FixedTimeStep::Interpolation::Reset()
{
    for (PoolTracker& tracker : s_pools)
        tracker.slots.clear();
    s_camera = TrackedSlot{};
    s_anythingApplied = false;
}

// TEMP (development only): what transform is a seated ped actually drawn with, and how many times per frame
void FixedTimeStep::DebugOnEntityRender(CEntitySAInterface* entity)
{
    if (!(FixedTimeStep::GetDebugFlags() & DEBUG_DUMP_CAMERA) || !entity || entity->nType != 3 /* ENTITY_TYPE_PED */ || !entity->matrix)
        return;

    auto* ped = reinterpret_cast<CPedSAInterface*>(entity);
    if (!ped->pedFlags.bInVehicle || !ped->pVehicle)
        return;

    static unsigned long       lastDump = 0;
    static unsigned int        rendersSinceDump = 0;
    static CEntitySAInterface* lastPed = nullptr;
    rendersSinceDump++;

    const unsigned long now = GetTickCount32();
    if (now - lastDump < 1000)
        return;

    auto ltmPos = [](CEntitySAInterface* e)
    {
        if (!e->m_pRwObject)
            return e->matrix ? e->matrix->vPos : e->m_transform.m_translate;
        RwMatrix* ltm = reinterpret_cast<RwMatrix*(__cdecl*)(void*)>(0x7F0990)(e->m_pRwObject->object.parent);  // RwFrameGetLTM
        return CVector(ltm->pos.x, ltm->pos.y, ltm->pos.z);
    };

    auto*          veh = reinterpret_cast<CEntitySAInterface*>(ped->pVehicle);
    const CVector  pedLtm = ltmPos(entity);
    const CVector  vehLtm = ltmPos(veh);
    const CVector& pedMat = entity->matrix->vPos;
    const CVector  vehMat = veh->matrix ? veh->matrix->vPos : veh->m_transform.m_translate;
    g_pCore->GetConsole()->Printf("[fts] render ped: renders/s=%u | pedLtm-vehLtm %.3f | pedLtm-pedMat %.3f | vehLtm-vehMat %.3f | pedMat-vehMat %.3f",
                                  rendersSinceDump, (pedLtm - vehLtm).Length(), (pedLtm - pedMat).Length(), (vehLtm - vehMat).Length(),
                                  (pedMat - vehMat).Length());
    rendersSinceDump = 0;
    lastDump = now;
    lastPed = entity;
}
