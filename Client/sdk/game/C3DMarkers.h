/*****************************************************************************
 *
 *  PROJECT:     Multi Theft Auto v1.0
 *  LICENSE:     See LICENSE in the top level directory
 *  FILE:        sdk/game/C3DMarkers.h
 *  PURPOSE:     3D marker manager interface
 *
 *  Multi Theft Auto is available from https://www.multitheftauto.com/
 *
 *****************************************************************************/

#pragma once
#include "enums/T3DMarkerType.h"

class C3DMarker;
class CVector;

class C3DMarkers
{
public:
    virtual C3DMarker* CreateMarker(DWORD Identifier, T3DMarkerType::Enum dwType, CVector* vecPosition, float fSize, float fPulseFraction, BYTE r, BYTE g,
                                    BYTE b, BYTE a) = 0;
    virtual C3DMarker* FindFreeMarker() = 0;
    virtual C3DMarker* FindMarker(DWORD Identifier) = 0;
    virtual void       ReinitMarkers() = 0;
    // C3dMarkers::Update: activates the markers placed since the last render. The game runs it once per
    // simulation step; with the fixed time step it also has to run before every rendered frame.
    virtual void       Update() = 0;
};
