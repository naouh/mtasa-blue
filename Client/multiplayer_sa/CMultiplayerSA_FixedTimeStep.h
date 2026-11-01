/*****************************************************************************
 *
 *  PROJECT:     Multi Theft Auto
 *  LICENSE:     See LICENSE in the top level directory
 *  FILE:        multiplayer_sa/CMultiplayerSA_FixedTimeStep.h
 *
 *  Multi Theft Auto is available from https://www.multitheftauto.com/
 *
 *****************************************************************************/
#pragma once

// Fixed time step simulation (world special property "fixedtimestep").
//
// GTA:SA advances its whole simulation once per rendered frame with a time step derived from the
// real frame time, which makes vehicle handling, jumps and collisions depend on the frame rate.
// When this mode is enabled the simulation runs at a fixed rate instead: each rendered frame
// performs 0..N simulation steps of exactly 1000/rate milliseconds, so the physics behaves as if
// the game was capped at `rate` FPS no matter how fast frames are actually rendered.
//
// Entry points used by the existing hooks in CMultiplayerSA_Rendering.cpp (CTimer::Update) and
// CMultiplayerSA.cpp (Idle -> CGame::Process).
namespace FixedTimeStep
{
    // Called at the start of CTimer::Update, before GTA samples the clock.
    // Decides how many simulation steps the frame gets and primes CTimer so that the step it is
    // about to compute has exactly the fixed length (or zero length when no step runs this frame).
    void OnTimerUpdate();

    // Replaces the single CGame::Process call of Idle: runs the number of steps decided above.
    void ProcessGame(void(__cdecl* CGame_Process)());

    // Places the world where it is displayed for this frame. Called by the client after its idle pulses, so that
    // everything between this and RestoreWorld is the render phase and everything else works on the true world.
    void ApplyRender();

    // Puts the true transforms back. Called by the client before its post-frame pulses: the network code must send, and
    // sync corrections must start from, the simulation's state rather than a displayed prediction.
    void RestoreWorld();

    // TEMP (development only), see CMultiplayer::SetFixedTimeStepDebugFlags
    enum DebugFlags : unsigned int
    {
        DEBUG_NO_CAMERA = 1,
        DEBUG_NO_ENTITIES = 2,
        DEBUG_NO_TIMESTEP_PIN = 4,
        DEBUG_VEHICLES_ONLY = 8,
        DEBUG_EXTRAPOLATE = 16,              // project the last step forward instead of blending back towards the previous one
        DEBUG_DUMP_CAMERA = 32,              // print the CCamera matrix and the RW camera frame to the console once per second
        DEBUG_CAMERA_DERIVED_VALUES = 64,    // also recompute CCamera's frustum/view (CalculateDerivedValues) after moving the camera
        DEBUG_LERP_SEATED_PEDS = 128,        // interpolate seated peds like any entity instead of re-seating them from the vehicle
        DEBUG_EXTRAPOLATE_CAMERA = 256,      // with DEBUG_EXTRAPOLATE: extrapolate the camera like an entity (default: anchored to its displayed target)
        DEBUG_INTERPOLATE_CAMERA_POS = 512,  // unused for now (kept so flag values stay stable)
        DEBUG_ATTACH_CAMERA = 1024,          // experiment: camera carried in its target's frame (relative pose interpolated)
    };
    unsigned int GetDebugFlags();
    void         DebugOnEntityRender(CEntitySAInterface* entity);  // TEMP (development only), called from CEntity::Render

    // Render-side interpolation of pooled entities and the camera between two simulation steps,
    // see CMultiplayerSA_FixedTimeStepInterpolation.cpp
    namespace Interpolation
    {
        void Capture();  // after a simulation step: remember the new transforms
        void Apply(float alpha, float cameraPosAlpha,
                   float cameraRotAlpha);  // before rendering: entities at `alpha`, camera position and orientation at their own alphas
        void Reapply();                    // before onClientRender: put this frame's displayed transforms back after Restore
        void Restore();                    // after the frame is drawn: put the true transforms back
        void AbsorbExternalChanges();      // before a step: take in what MTA wrote since the last capture (sync corrections)
        void Reset();                      // forget all tracked state
    }  // namespace Interpolation
}  // namespace FixedTimeStep
