//-------------------------------------------------------------------------
// LLMapper autonomous Blood playtest bot.
//-------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "fix16.h"
#include "controls.h"

class LLMapperBot
{
public:
    LLMapperBot();
    ~LLMapperBot();

    void Enable(const char *telemetry, const char *trajectory, const char *demo);
    void ConfigureTimeout(int seconds);
    void ConfigureStallTimeout(int seconds);
    void SetFast(bool fast);
    void SetVisible(bool visible);

    bool Enabled() const { return m_enabled; }
    bool Fast() const { return m_fast; }
    bool Visible() const { return m_visible; }

    // Called once after resources and the demo subsystem are initialized.
    void PrepareLaunch();
    // Called from ctrlGetInput. The returned GINPUT goes through the normal
    // network FIFO, playerProcess(), and CDemo::Write() path.
    GINPUT GetInput();
    // Called after one real ProcessFrame() has completed.
    void OnFrame();
    // Called by the normal level-exit event path.
    void OnLevelExit(int exitType);
    // Called during shutdown/restart to flush telemetry and the demo.
    void Finish(const char *reason = nullptr);

private:
    LLMapperBot(const LLMapperBot &) = delete;
    LLMapperBot &operator=(const LLMapperBot &) = delete;

    struct Impl;
    Impl *m_impl;
    bool m_enabled;
    bool m_fast;
    bool m_visible;
};

extern LLMapperBot gLLMapperBot;
