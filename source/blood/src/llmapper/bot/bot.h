//-------------------------------------------------------------------------
// Minimal autonomous LLMapper Blood bot integration.
//-------------------------------------------------------------------------
#pragma once

#include <cstdint>

#include "fix16.h"
#include "../../controls.h"

struct LLMapperPlayerState
{
    int x = 0;
    int y = 0;
    int z = 0;
    int sector = -1;
    int angle = 0;
    int horizon = 0;
    bool onGround = false;
    bool crouched = false;
    bool alive = false;
};

class LLMapperBot
{
public:
    LLMapperBot();
    ~LLMapperBot();

    void Enable(const char *telemetry, const char *trajectory, const char *demo);
    void ConfigureTimeout(int seconds);
    void SetFast(bool fast);
    void SetVisible(bool visible);

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
