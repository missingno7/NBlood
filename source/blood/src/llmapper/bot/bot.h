//-------------------------------------------------------------------------
// Minimal autonomous LLMapper Blood bot integration.
//-------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "fix16.h"
#include "../../controls.h"

class LLMapperBot
{
public:
    LLMapperBot();
    ~LLMapperBot();

    void Enable(const char *telemetry, const char *trajectory, const char *demo);
    void ConfigureTimeout(int seconds);
    void SetFast(bool fast);
    void SetVisible(bool visible);
    // Paint the semantic world into the game view. Independent of the bot:
    // it is for looking at the model while a person plays the level.
    void SetDebugOverlay(bool debug);
    bool DebugOverlay() const { return m_debug; }

    bool Enabled() const { return m_enabled; }
    bool Fast() const { return m_fast; }
    bool Visible() const { return m_visible; }
    void PrepareLaunch();
    GINPUT GetInput();
    void OnFrame();
    void OnActionResolved(int hit, int target, int extra, bool accepted, int key);
    void OnLevelExit(int exitType);
    void Finish(const char *reason = nullptr);
    void DrawStatus();
    void DrawDebugOverlay(int cameraX, int cameraY, int cameraZ,
                          fix16_t cameraAngle, fix16_t cameraHorizon);

private:
    LLMapperBot(const LLMapperBot &) = delete;
    LLMapperBot &operator=(const LLMapperBot &) = delete;

    struct Impl;
    Impl *m_impl;
    bool m_enabled;
    bool m_fast;
    bool m_visible;
    bool m_debug;
};

extern LLMapperBot gLLMapperBot;
