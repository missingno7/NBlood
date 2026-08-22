//-------------------------------------------------------------------------
// LLMapper autonomous Blood playtest bot.
//-------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "fix16.h"
// bot.h is included both from blood/src and its own nested directory.
#include "../../controls.h"

class LLMapperBot
{
public:
    LLMapperBot();
    ~LLMapperBot();

    void Enable(const char *telemetry, const char *trajectory, const char *demo);
    void ConfigureTimeout(int seconds);
    void ConfigureStallTimeout(int seconds);
    void ConfigureNavMeshDump(const char *path);
    bool ConfigureDebugOverlay(const char *setting);
    void DescribeDebugOverlay(char *buffer, int size) const;
    void SetFast(bool fast);
    void SetVisible(bool visible);

    bool Enabled() const { return m_enabled; }
    bool Fast() const { return m_fast; }
    bool Visible() const { return m_visible; }
    bool DebugOverlayEnabled() const { return m_debugOverlay; }

    // Called once after resources and the demo subsystem are initialized.
    void PrepareLaunch();
    // Called from ctrlGetInput. The returned GINPUT goes through the normal
    // network FIFO, playerProcess(), and CDemo::Write() path.
    GINPUT GetInput();
    // Called after one real ProcessFrame() has completed.
    void OnFrame();
    // Called by ProcessInput after the real engine action scan and trigger
    // dispatch have resolved a gameplay USE pulse.
    void OnActionResolved(int hit, int target, int extra, bool accepted, int key);
    // Called from the authoritative player damage path with the engine source.
    void OnBotDamaged(int source, int damageType, int amount);
    // Called after the weapon's authoritative VectorScan so ranged
    // activation failures can report the object actually hit.
    void OnVectorResolved(int hit, int sector, int wall, int sprite,
                          int x, int y, int z);
    // Called by the normal level-exit event path.
    void OnLevelExit(int exitType);
    // Called during shutdown/restart to flush telemetry and the demo.
    void Finish(const char *reason = nullptr);
    // Draws a short status readout in a visible run.  Times shown here match
    // the game_time field in the telemetry, so a moment seen on screen can
    // be looked up directly in the run log.
    void DrawStatus();
    void DrawDebugOverlay(int cameraX, int cameraY, int cameraZ,
                          fix16_t cameraAngle, fix16_t cameraHoriz);

private:
    LLMapperBot(const LLMapperBot &) = delete;
    LLMapperBot &operator=(const LLMapperBot &) = delete;

    struct Impl;
    Impl *m_impl;
    bool m_enabled;
    bool m_fast;
    bool m_visible;
    bool m_debugOverlay;
};

extern LLMapperBot gLLMapperBot;
