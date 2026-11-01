/*****************************************************************************
 *
 *  PROJECT:     Multi Theft Auto
 *  LICENSE:     See LICENSE in the top level directory
 *  FILE:        multiplayer_sa/CMultiplayerSA_FixedTimeStep.cpp
 *
 *  Multi Theft Auto is available from https://www.multitheftauto.com/
 *
 *****************************************************************************/
#include "StdInc.h"
#include "CMultiplayerSA_FixedTimeStep.h"
#include <algorithm>
#include <cstdint>

// How GTA:SA computes its time step (CTimer::Update, 0x561B10):
//
//   before = m_snRenderStartTime
//   m_snRenderStartTime = ms_fnTimerFunction()            // raw performance counter ticks
//   delta = (m_snRenderStartTime - before) * ms_fTimeScale // unless paused
//   frameMs = delta / m_snTimerDivider                     // m_snTimerDivider = ticks per millisecond
//   m_snTimeInMilliseconds += (uint32)min(frameMs, 300)
//   ms_fTimeStep = clamp(max(frameMs / 20, 0.01), 0.00001, 3)
//
// We do not patch any of that. Instead, right before CTimer::Update samples the clock, we move
// m_snRenderStartTime so that the delta it computes is exactly one fixed step (or ~zero on frames
// that do not simulate). Everything downstream (time step, game clock, frame counter, the frame
// rate fixes hooked at the end of CTimer::Update) then behaves exactly as it does when the game is
// capped at `rate` FPS. ms_fnTimerFunction itself is left untouched because CStreaming uses it for
// its per-frame loading budget and must keep seeing real time.
namespace
{
    using TimerFunction_t = std::uint64_t(__cdecl*)();

    constexpr std::uintptr_t ADDR_CTimer_ms_fnTimerFunction = 0xB7CB28;
    constexpr std::uintptr_t ADDR_CTimer_m_snTimerDivider = 0xB7CB2C;
    constexpr std::uintptr_t ADDR_CTimer_m_snRenderStartTime = 0xB7CB38;
    constexpr std::uintptr_t ADDR_CTimer_m_CodePause = 0xB7CB48;
    constexpr std::uintptr_t ADDR_CTimer_m_UserPause = 0xB7CB49;
    constexpr std::uintptr_t ADDR_CTimer_ms_fTimeStepNonClipped = 0xB7CB58;
    constexpr std::uintptr_t ADDR_CTimer_ms_fTimeStep = 0xB7CB5C;
    constexpr std::uintptr_t ADDR_CTimer_ms_fTimeScale = 0xB7CB64;
    constexpr std::uintptr_t FUNC_CTimer_Update = 0x561B10;

    // CTimer::UpdateVariables keeps ms_fTimeStep inside these bounds while the game runs; the render
    // path (door swing, cloud drift, anim blending...) relies on the floor and would blow up below it.
    constexpr float RUNNING_MIN_TIME_STEP = 0.01f;
    constexpr float MAX_TIME_STEP = 3.0f;
    constexpr float TIME_STEP_UNIT_MS = 20.0f;  // ms_fTimeStep == 1.0 is 1/50 s

    // Upper bound on catch-up work after a slow frame. Beyond this the game slows down instead of
    // running ever more steps per frame (the "spiral of death").
    constexpr int MAX_STEPS_PER_FRAME = 4;

    // Real time beyond this per frame is dropped (a stall, alt-tab, loading). CTimer clips its own
    // millisecond clock at 300ms, we are stricter because every clipped millisecond costs a step.
    constexpr double MAX_FRAME_MS = 100.0;

    bool          s_enabled = false;
    float         s_rate = 50.0f;  // one GTA time step unit == 50 steps per second
    double        s_accumulatorTicks = 0.0;
    std::uint64_t s_lastRealTicks = 0;
    bool          s_hasLastRealTicks = false;
    int           s_stepsThisFrame = -1;       // -1: this frame is not driven by us (pass-through)
    bool          s_extraStepPending = false;  // set while we re-enter CTimer::Update for steps 2..N
    unsigned int  s_debugFlags = 0;
    double        s_frameDeltaMs = 20.0;  // real time of the last rendered frame, what the render path is told
    int           s_lastFrameSteps = -1;  // steps run by the frame being rendered (-1: pass-through frame)
    bool          s_renderPending = false;  // ProcessGame ran in fixed mode, ApplyRender has yet to place the world
    bool          s_renderApplied = false;  // ApplyRender already placed the world for this frame: the next call re-applies it

    TimerFunction_t GetTimerFunction()
    {
        return *reinterpret_cast<TimerFunction_t*>(ADDR_CTimer_ms_fnTimerFunction);
    }
    std::uint64_t& RenderStartTime()
    {
        return *reinterpret_cast<std::uint64_t*>(ADDR_CTimer_m_snRenderStartTime);
    }
    double TicksPerMs()
    {
        return static_cast<double>(*reinterpret_cast<std::uint32_t*>(ADDR_CTimer_m_snTimerDivider));
    }
    bool IsGamePaused()
    {
        return *reinterpret_cast<bool*>(ADDR_CTimer_m_CodePause) || *reinterpret_cast<bool*>(ADDR_CTimer_m_UserPause);
    }
    std::uint64_t StepTicks()
    {
        return static_cast<std::uint64_t>(TicksPerMs() * 1000.0 / s_rate);
    }

    void ResetAccumulation()
    {
        s_accumulatorTicks = 0.0;
        s_hasLastRealTicks = false;
        s_stepsThisFrame = -1;
        s_extraStepPending = false;
        s_renderPending = false;
    }
}  // namespace

void FixedTimeStep::OnTimerUpdate()
{
    // Whatever was rendered interpolated last frame goes back to its true transform before anything
    // (simulation or not) looks at the world again. Cheap no-op when nothing was interpolated.
    if (!s_extraStepPending)
    {
        Interpolation::Restore();
        // What the post-frame pulses wrote must not count as motion of the coming step
        if (s_enabled)
            Interpolation::AbsorbExternalChanges();
    }

    if (!s_enabled)
    {
        ResetAccumulation();
        return;
    }

    const TimerFunction_t timerFunction = GetTimerFunction();
    if (!timerFunction)
    {
        // CTimer::Initialise has not run yet, CTimer::Update returns early in that case too
        ResetAccumulation();
        return;
    }

    const std::uint64_t now = timerFunction();
    const std::uint64_t stepTicks = StepTicks();

    if (s_extraStepPending)
    {
        // Re-entered from ProcessGame for an additional step this frame: make it a full step as well
        s_extraStepPending = false;
        RenderStartTime() = now - stepTicks;
        return;
    }

    if (IsGamePaused())
    {
        // CTimer zeroes the delta while paused and CGame::Process must keep running once per frame
        // (it is what processes the un-pause input), so let the frame through untouched.
        ResetAccumulation();
        return;
    }

    double deltaTicks = s_hasLastRealTicks ? static_cast<double>(now - s_lastRealTicks) : static_cast<double>(stepTicks);
    s_lastRealTicks = now;
    s_hasLastRealTicks = true;

    deltaTicks = std::min(deltaTicks, MAX_FRAME_MS * TicksPerMs());
    s_accumulatorTicks += deltaTicks;
    s_frameDeltaMs = deltaTicks / TicksPerMs();

    int steps = static_cast<int>(s_accumulatorTicks / static_cast<double>(stepTicks));
    steps = std::min(steps, MAX_STEPS_PER_FRAME);
    s_accumulatorTicks -= static_cast<double>(steps) * static_cast<double>(stepTicks);
    s_stepsThisFrame = steps;

    // CTimer::Update samples the clock a few hundred nanoseconds after us, so the delta it sees is
    // stepTicks plus that tiny bias, never less. The bias is far below one millisecond and does not
    // reach the integer game clock.
    RenderStartTime() = (steps > 0) ? (now - stepTicks) : now;
}

void FixedTimeStep::ProcessGame(void(__cdecl* CGame_Process)())
{
    const int steps = s_stepsThisFrame;
    s_stepsThisFrame = -1;
    s_lastFrameSteps = steps;
    s_renderPending = false;

    if (steps < 0)
    {
        CGame_Process();
        return;
    }

    for (int i = 0; i < steps; ++i)
    {
        if (i > 0)
        {
            // Idle only calls CTimer::Update once per frame; advance the game clock ourselves for
            // the extra steps so each CGame::Process sees a fresh, full-length time step.
            s_extraStepPending = true;
            reinterpret_cast<void(__cdecl*)()>(FUNC_CTimer_Update)();
        }
        CGame_Process();
        Interpolation::Capture();
    }

    // The simulation saw the fixed step; the render path now sees the real frame time, exactly as in
    // the unmodified game. Per-frame effects (cloud drift, fog, door swing, anim blending) therefore
    // run at their normal speed and stay smooth, instead of being frozen between steps or stepping 50 times a second.
    if (!(s_debugFlags & DEBUG_NO_TIMESTEP_PIN))
    {
        const float timeScale = *reinterpret_cast<float*>(ADDR_CTimer_ms_fTimeScale);
        const float frameStep = std::clamp(static_cast<float>(s_frameDeltaMs) * timeScale / TIME_STEP_UNIT_MS, RUNNING_MIN_TIME_STEP, MAX_TIME_STEP);
        *reinterpret_cast<float*>(ADDR_CTimer_ms_fTimeStep) = frameStep;
        *reinterpret_cast<float*>(ADDR_CTimer_ms_fTimeStepNonClipped) = frameStep;
    }

    s_renderPending = true;
    s_renderApplied = false;
}

void FixedTimeStep::ApplyRender()
{
    // Called twice per frame: before the render and again before onClientRender, after the network code ran on the
    // restored world. The accumulator has not moved in between, so both calls place the world at the same alpha.
    if (!s_renderPending)
        return;
    if (s_renderApplied)
    {
        Interpolation::Reapply();
        return;
    }
    s_renderApplied = true;

    // Place the world where it is "now", between the last step and the next one, for this frame's render
    const double stepTicks = static_cast<double>(StepTicks());
    float        alpha = stepTicks > 0.0 ? static_cast<float>(std::min(s_accumulatorTicks / stepTicks, 1.0)) : 0.0f;
    // Interpolation shows the world one step late (prev -> cur). Extrapolation shows it where the last
    // step's motion puts it now (cur -> cur + (cur - prev) * alpha), no display latency, at the price of a
    // one-step overshoot on sudden stops.
    // The camera keeps interpolating by default even then: its orientation follows the mouse, which no
    // linear projection predicts, and an extrapolated camera corrects itself at every step (visible jitter
    // when looking around while driving).
    // Its position however must share the entities' time reference, or the vehicle it follows jitters in frame.
    // Camera: one alpha. 16 = interpolated camera, 16+256 = extrapolated camera (same reference as the entities).
    // The attached-camera experiment (position and orientation carried in the target's frame) is behind 1024.
    float cameraPosAlpha = alpha;
    float cameraRotAlpha = alpha;
    if (s_debugFlags & DEBUG_EXTRAPOLATE)
    {
        alpha += 1.0f;
        if (s_debugFlags & DEBUG_EXTRAPOLATE_CAMERA)
        {
            cameraPosAlpha += 1.0f;
            cameraRotAlpha += 1.0f;
        }
    }
    Interpolation::Apply(alpha, cameraPosAlpha, cameraRotAlpha);
}

void FixedTimeStep::RestoreWorld()
{
    Interpolation::Restore();
}

void CMultiplayerSA::ApplyFixedTimeStepRender()
{
    FixedTimeStep::ApplyRender();
}

void CMultiplayerSA::RestoreFixedTimeStepWorld()
{
    FixedTimeStep::RestoreWorld();
}

unsigned int FixedTimeStep::GetDebugFlags()
{
    return s_debugFlags;
}

void CMultiplayerSA::SetFixedTimeStepDebugFlags(unsigned int flags)
{
    FixedTimeStep::Interpolation::Restore();
    s_debugFlags = flags;
}

bool CMultiplayerSA::IsFixedTimeStepEnabled() const noexcept
{
    return s_enabled;
}

bool CMultiplayerSA::DidFixedTimeStepRunThisFrame() const noexcept
{
    return !s_enabled || s_lastFrameSteps != 0;
}

void CMultiplayerSA::SetFixedTimeStepEnabled(bool enabled)
{
    if (s_enabled == enabled)
        return;

    s_enabled = enabled;
    ResetAccumulation();

    if (!enabled)
    {
        // May be called mid-frame from a script while the world is rendered interpolated
        FixedTimeStep::Interpolation::Restore();
        FixedTimeStep::Interpolation::Reset();
    }
}

float CMultiplayerSA::GetFixedTimeStepRate() const noexcept
{
    return s_rate;
}

bool CMultiplayerSA::SetFixedTimeStepRate(float rate)
{
    if (!(rate >= FIXED_TIME_STEP_MIN_RATE && rate <= FIXED_TIME_STEP_MAX_RATE))
        return false;

    s_rate = rate;
    ResetAccumulation();
    return true;
}
