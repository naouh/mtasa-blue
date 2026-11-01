/*****************************************************************************
 *
 *  PROJECT:     Multi Theft Auto v1.0
 *  LICENSE:     See LICENSE in the top level directory
 *  FILE:        sdk/game/CCheckpoints.h
 *  PURPOSE:     Checkpoint entity manager interface
 *
 *  Multi Theft Auto is available from https://www.multitheftauto.com/
 *
 *****************************************************************************/

#pragma once

class CCheckpoint;
class CVector;

class CCheckpoints
{
public:
    virtual CCheckpoint* CreateCheckpoint(DWORD Identifier, WORD wType, CVector* vecPosition, CVector* vecPointDir, float fSize, float fPulseFraction,
                                          const SharedUtil::SColor color) = 0;
    virtual CCheckpoint* FindFreeMarker() = 0;
    virtual CCheckpoint* FindMarker(DWORD identifier) = 0;
    // CCheckpoints::Update: places the 3D marker of every live checkpoint (and rotates it). The game runs
    // it once per simulation step; with the fixed time step it also has to run before every rendered frame.
    virtual void         Update() = 0;
};
