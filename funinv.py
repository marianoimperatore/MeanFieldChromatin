#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 15:40:11 2022

@author: mariano
"""




import sys, os

import cooler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as scy
import time

import pdb
import esprenv as env

from numba import njit, vectorize

# settings
N = 170
R = 3
NR = N * R
M = 5
# Nmax = 9





def buildP( bb):
    PP = np.zeros((N,N))
    for i in range(N):
         for j in range(N):
             FFt = 0
             Ri, Rj = i * R, j * R
             for ri in range(R):
                 for rj in range(R):
                     if (bb[ Ri + ri ] == 0) | (bb[ Rj + rj ] == 0) :
                         FFt = FFt + FF[ 0, np.abs( Ri + ri - (Rj + rj) )] 
                     else :
                         FFt = FFt + FF[ np.int_( bb[ Ri + ri ] == bb[ Rj + rj ]), np.abs( Ri + ri - (Rj + rj) )] 
                    
    
             PP[ i, j] = FFt # /(R**2)    

    # PP = np.triu(PP,0) + np.triu(PP,1).T
    return PP



def buildP2( bb, ppc, ppo, Rir):
    ppi = np.where( (bb[None,:] == bb[:, None]) & (bb[None,:] != 0) & (bb[:,None] != 0), ppc, ppo )
    
    ppir = np.add.reduceat( np.add.reduceat( ppi, Rir), Rir, 1)    
   
    return ppir




def buildPtriu( bb):
    PP = np.zeros((N,N))
    for i in range(N):
         for j in range(i+1,N):
             FFt = 0
             Ri, Rj = i * R, j * R
             for ri in range(R):
                 for rj in range(R):
                     if (bb[ Ri + ri ] == 0) | (bb[ Rj + rj ] == 0) :
                         FFt = FFt + FF[ 0, np.abs( Ri + ri - (Rj + rj) )] 
                     else :
                         FFt = FFt + FF[ np.int_( bb[ Ri + ri ] == bb[ Rj + rj ]), np.abs( Ri + ri - (Rj + rj) )] 
                    
    
             PP[ i, j] = FFt # /(R**2)    

    # PP = np.triu(PP,0) + np.triu(PP,1).T
    return PP



def buildPnodiag( bb):
    PP = np.zeros((N,N))
    for i in range(N):
         for j in range(i+1,N):
             FFt = 0
             Ri, Rj = i * R, j * R
             for ri in range(R):
                 for rj in range(R):
                     if (bb[ Ri + ri ] == 0) | (bb[ Rj + rj ] == 0) :
                         FFt = FFt + FF[ 0, np.abs( Ri + ri - (Rj + rj) )] 
                     else :
                         FFt = FFt + FF[ np.int_( bb[ Ri + ri ] == bb[ Rj + rj ]), np.abs( Ri + ri - (Rj + rj) )] 
                    
    
             PP[ i, j] = FFt # /(R**2)    

    PP = np.triu(PP,1) + np.triu(PP,1).T
    return PP














# =============================================================================
# Run vectorize
# =============================================================================
@njit(fastmath=True)
def calcDPeff(x1,xx1,TTcn):
    
    coln = TTcn[ x1  ]
    colo = TTc[ x1 ]
    DP1 = np.zeros((N))
    
    if coln == 0 :
        for ni in range(N):
            for ri in range(R):
                DP1[ni] = DP1[ni] + FF[ 0, np.abs( x1 - (ni * R + ri) )] - \
                    FF[ np.int_( colo ==TTc[ ni * R + ri ]), np.abs( x1 - (ni * R + ri) )]    
        
    elif colo == 0 :
        for ni in range(N):
            for ri in range(R):
                DP1[ni] = DP1[ni] + FF[ np.int_( coln ==TTcn[ ni * R + ri ]), np.abs( x1 - (ni * R + ri) )] - \
                    FF[ 0, np.abs( x1 - (ni * R + ri) )]
                    
    else:
        for ni in range(N):
            for ri in range(R):
                DP1[ni] = DP1[ni] + FF[ np.int_( coln ==TTcn[ ni * R + ri ]), np.abs( x1 - (ni * R + ri) )] - \
                    FF[ np.int_( colo ==TTc[ ni * R + ri ]), np.abs( x1 - (ni * R + ri) )]
    
        
    # self interacting square        
    if False:
        for ri1 in range(R):
            for ri2 in range(R):
                DP1[xx1] = DP1[xx1] + FF[ np.int_( TTcn[ x1 + ri1 ] ==TTcn[ x1 + ri2 ]), np.abs( x1 + ri1 - (x1 + ri2) )] - \
                    FF[ np.int_( TTc[ x1 + ri1 ] ==TTc[ x1 + ri2 ]), np.abs( x1 + ri1 - (x1 + ri2) )]        
        
        
    return  DP1






@njit(fastmath=True)
def DcostFeff( P, DP, alp, Dalp, F, DF, xx1):
    
    sCR2DP = 2*(CR2[xx1]*DP).sum()
    
    DH = 1/N2R4 * (
        (alp**2 + Dalp * (Dalp + 2*alp) ) * ( 2*(DP**2).sum() + 2*2*( P[xx1] * DP).sum() )  # 1
            + Dalp * (Dalp + 2*alp) * (P**2).sum()  # 1
                + Dalp * (Dalp - 2*(1-alp)) * Po2  # 2
                    + DF * (DF + 2*F) * sCR2   # 3
                        + 2*Dalp*(1-2*alp - Dalp) * ((P*Po).sum() + 2*(DP*Po[xx1]).sum())  # 4
                            + 2*alp*(1-alp)* 2*(DP*Po[xx1]).sum()  # 4
                                - 2 * (Dalp*DF + Dalp*F + alp*DF)* (sCR2P + sCR2DP) # 5
                                    -2 * alp*F* sCR2DP  # 5
                                        - 2*(DF*(1-alp-Dalp) - F*Dalp) * sCR2Po # 6
        )
    
    return DH



@njit(fastmath=True)
def DcostFeff2( P, DP, alp, Dalp, F, DF, xx1):
    
    DH = 1/N2R4 * (
        (alp**2 + Dalp * (Dalp + 2*alp) ) * ( 2*(DP**2).sum() + 2*2*( P[xx1] * DP).sum() )  # 1
            + Dalp * (Dalp + 2*alp) * (P**2).sum()  # 1
                + Dalp * (Dalp - 2*(1-alp)) * (Po**2).sum()  # 2
                    + DF * (DF + 2*F) * (CR2**2).sum()   # 3
                        + 2*Dalp*(1-2*alp - Dalp) * ((P * Po).sum() + 2*(DP*Po[xx1]).sum())  # 4
                            + 2*alp*(1-alp)* 2*(DP*Po[xx1]).sum()  # 4
                                - 2 * (Dalp*DF + Dalp*F + alp*DF)*((CR2*P).sum()+2*(CR2[xx1]*DP).sum()) # 5
                                    -2 * alp*F* 2*(CR2[xx1]*DP).sum()  # 5
                                        - 2*(DF*(1-alp-Dalp) - F*Dalp) * (CR2*Po).sum() # 6
        )
    
    return DH


@njit(fastmath=True)
def CostFeff( Pn, alp, F):
    
    H = 1/N2R4 * np.sum( (alp * Pn + (1-alp) * Po - F * CR2)**2  )

    return H



















# =============================================================================
# Plot
# =============================================================================
def plMaps( Pi, CR2, F):

    # Pf = buildPnodiag( ttc)
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    
    axs[0].imshow( Pi, cmap='hot' )
    axs[1].imshow( CR2, cmap='hot' )
    # plt.figure()
    # plt.imshow( np.log(np.abs((Pi - F*CR2)/Pi)), cmap='hot' )
    plt.savefig('plMaps.png')




def plProfile( NR, tcc, F, Pi, Po, CR2):
    ########## compare s profiles
    tcc = np.ones((NR))
    Pc = buildPnodiag( tcc)
    
    Pfs = np.zeros((Pi.shape[0]))
    Pfss = np.zeros((Pi.shape[0]))
    
    Pos = np.zeros((Po.shape[0]))
    Poss = np.zeros((Po.shape[0]))
    
    Pcs = np.zeros((Po.shape[0]))
    Pcss = np.zeros((Po.shape[0]))
    
    CR2s = np.zeros((CR2.shape[0]))
    CR2ss = np.zeros((CR2.shape[0]))
    
    
    for idiag in range( 0, Pi.shape[0]):
        cmdiag = np.diagonal( Pi, idiag)
        Pfs[idiag] = cmdiag.mean() 
        Pfss[idiag] = cmdiag.std() 
    
        cmdiag = np.diagonal( Po, idiag)
        Pos[idiag] = cmdiag.mean() 
        Poss[idiag] = cmdiag.std() 
    
        cmdiag = np.diagonal( Pc, idiag)
        Pcs[idiag] = cmdiag.mean() 
        Pcss[idiag] = cmdiag.std() 
    
        cmdiag = np.diagonal( CR2, idiag)
        CR2s[idiag] = cmdiag.mean() 
        CR2ss[idiag] = cmdiag.std() 
    
    
    plt.figure(figsize=(6,3))
    plt.plot( Pos[1:], 'k--')
    plt.plot( Pcs[1:], 'k-')
    plt.plot( F * CR2s[1:], 'bo')
    plt.plot( Pfs[1:], 'r.')
    plt.legend(['Open','Closed','Cs','Pfs'])
    plt.semilogy()
    plt.title('Genomic distance maps profiles')
    plt.savefig('plProfile.png')
    
    plt.figure(figsize=(6,3))
    plt.plot( np.abs(Pfs - F * CR2s)[1:], '.')
    plt.legend(['Pfs - Cs'])
    plt.semilogy()
    plt.title('Absolute error')
    plt.savefig('plProfileError.png')




# =============================================================================
# Correlations
# =============================================================================
def corrMaps( CR2, Pi):
    corrm = pd.DataFrame(
        data = {'Cs': CR2.flatten(), 'Pf': Pi.flatten()}
        ).corr( method='pearson')
    corrmsp = pd.DataFrame(
        data = {'Cs': CR2.flatten(), 'Pf': Pi.flatten()}
        ).corr( method='spearman')    
    
    
    CR2m = np.copy(CR2)
    Pfsm = np.copy(Pi)
    
    # calc diag
    Pfs = np.zeros((Pi.shape[0]))
    Pfss = np.zeros((Pi.shape[0]))
    
    CR2s = np.zeros((CR2.shape[0]))
    CR2ss = np.zeros((CR2.shape[0]))
    
    
    for idiag in range( 0, Pi.shape[0]):
        cmdiag = np.diagonal( Pi, idiag)
        Pfs[idiag] = cmdiag.mean() 
        Pfss[idiag] = cmdiag.std() 
    
        cmdiag = np.diagonal( CR2, idiag)
        CR2s[idiag] = cmdiag.mean() 
        CR2ss[idiag] = cmdiag.std() 
        
        
    # remove diag
    for di1 in range(CR2.shape[0]):
        for di2 in range(CR2.shape[1]):
            CR2m[di1,di2] -= CR2s[ np.abs(di1-di2)]
            Pfsm[di1,di2] -= Pfs[ np.abs(di1-di2)]
            
    corrs = pd.DataFrame(
        data = {'Cs': CR2m.flatten(), 'Pf': Pfsm.flatten()}
        ).corr( method='pearson')
    
    corrssp = pd.DataFrame(
        data = {'Cs': CR2m.flatten(), 'Pf': Pfsm.flatten()}
        ).corr( method='spearman')    
    
    print('corrm:', '%.2f' % corrm.iloc[0,1],'corrs:','%.2f' % corrs.iloc[0,1],
          'corrmsp:','%.2f' % corrmsp.iloc[0,1],'corrssp:','%.2f' % corrssp.iloc[0,1])
    

    return pd.DataFrame(data={
        'corrm': [corrm.iloc[0,1]], 
        'corrs': [corrs.iloc[0,1]], 
        'corrmsp': [corrmsp.iloc[0,1]], 
        'corrssp': [corrssp.iloc[0,1]]
        })






def corrMaps2( CR2, Pi):
    forcorr = pd.DataFrame(
        data = {'Cs': CR2.flatten(), 'Pf': Pi.flatten()}
        )
    corrm = forcorr.corr( method='pearson')
    corrmsp = forcorr.corr( method='spearman')    
    
    
    CR2m = np.copy(CR2)
    Pfsm = np.copy(Pi)
    
    # calc diag
    Pfs = np.zeros((Pi.shape[0]))
    CR2s = np.zeros((CR2.shape[0]))
    
    
    for idiag in range( 0, Pi.shape[0]):
        cmdiag = np.diagonal( Pi, idiag)
        Pfs[idiag] = np.nanmean(cmdiag)
    
        cmdiag = np.diagonal( CR2, idiag)
        CR2s[idiag] = np.nanmean(cmdiag)
        
        
    # remove diag
    for di1 in range(CR2.shape[0]):
        for di2 in range(di1, CR2.shape[1]):
            CR2m[di1,di2] -= CR2s[ np.abs(di1-di2)]
            Pfsm[di1,di2] -= Pfs[ np.abs(di1-di2)]
            
            CR2m[di2,di1] = CR2m[di1,di2]
            Pfsm[di2,di1] = Pfsm[di1,di2]
            
       
    forcorr = pd.DataFrame(
        data = {'Cs': CR2m.flatten(), 'Pf': Pfsm.flatten()}
        )     
       
    corrs = forcorr.corr( method='pearson')
    corrssp = forcorr.corr( method='spearman')    
    

    return pd.DataFrame(data={
        'corrm': [corrm.iloc[0,1]], 
        'corrs': [corrs.iloc[0,1]], 
        'corrmsp': [corrmsp.iloc[0,1]], 
        'corrssp': [corrssp.iloc[0,1]]
        })







def corrMapsPea2( CR2, Pi):

    CR2m = np.copy(CR2)
    Pfsm = np.copy(Pi)
    
    # calc diag
    Pfs = np.zeros((Pi.shape[0]))
    CR2s = np.zeros((CR2.shape[0]))
    
    
    for idiag in range( 0, Pi.shape[0]):
        Pfs[idiag] = np.nanmean( np.diagonal( Pi, idiag) ) 
        CR2s[idiag] = np.nanmean( np.diagonal( CR2, idiag) )
        
        
    # remove diag
    for di1 in range(CR2.shape[0]):
        for di2 in range(CR2.shape[1]):
            CR2m[di1,di2] -= CR2s[ np.abs(di1-di2)]
            Pfsm[di1,di2] -= Pfs[ np.abs(di1-di2)]
    
    if False:
        corrm = np.corrcoef( CR2.flatten(), Pi.flatten())[0,1]
        corrs = np.corrcoef( CR2m.flatten(), Pfsm.flatten())[0,1]
    else:
        corrm = pd.DataFrame( {1:CR2.flatten(), 2:Pi.flatten()}).corr().values[0,1]
        corrs = pd.DataFrame( {1:CR2m.flatten(), 2:Pfsm.flatten()}).corr().values[0,1]
        

    return corrm, corrs





def makediag( PP):

    PPs = np.zeros((PP.shape[0]))
    PPss = np.zeros((PP.shape[0]))
    
    
    for idiag in range( 0, PP.shape[0]):
        cmdiag = np.diagonal( PP, idiag)
        PPs[idiag] = np.nanmean(cmdiag)
        PPss[idiag] = np.nanstd(cmdiag)
        
    return [PPs, PPss]










def plProfile2( ps, po, pc, yscale=['lin','log']):
    ########## compare s profiles
    tcc = np.ones((NR))
    Pc = buildPnodiag3( tcc)
    
    Pfs = np.zeros((Pi.shape[0]))
    Pfss = np.zeros((Pi.shape[0]))
    
    Pos = np.zeros((Po.shape[0]))
    Poss = np.zeros((Po.shape[0]))
    
    Pcs = np.zeros((Po.shape[0]))
    Pcss = np.zeros((Po.shape[0]))
    
    CR2s = np.zeros((CR2.shape[0]))
    CR2ss = np.zeros((CR2.shape[0]))
    
    
    for idiag in range( 0, Pi.shape[0]):
        cmdiag = np.diagonal( Pi, idiag)
        Pfs[idiag] = cmdiag.mean() 
        Pfss[idiag] = cmdiag.std() 
    
        cmdiag = np.diagonal( Po, idiag)
        Pos[idiag] = cmdiag.mean() 
        Poss[idiag] = cmdiag.std() 
    
        cmdiag = np.diagonal( Pc, idiag)
        Pcs[idiag] = cmdiag.mean() 
        Pcss[idiag] = cmdiag.std() 
    
        cmdiag = np.diagonal( CR2, idiag)
        CR2s[idiag] = cmdiag.mean() 
        CR2ss[idiag] = cmdiag.std() 
    
    # remove diag
    CR2m = np.copy(CR2)
    Pfsm = np.copy(Pi)    
    for di1 in range(CR2.shape[0]):
        for di2 in range(CR2.shape[1]):
            CR2m[di1,di2] -= CR2s[ np.abs(di1-di2)]
            Pfsm[di1,di2] -= Pfs[ np.abs(di1-di2)]    
    
    
    # Pf = buildPnodiag3( ttc)
    fig, axs = plt.subplots(2, 2, figsize=(12,12))
    
    if yscale[0] in ['log','loglog']:
        axs[0,0].imshow( np.log(CR2m), cmap='hot' )
        axs[1,0].imshow( np.log(Pfsm), cmap='hot' )    
    else:
        axs[0,0].imshow( CR2m, cmap='hot' )
        axs[1,0].imshow( Pfsm, cmap='hot' )    # plt.figure()
    # plt.imshow( np.log(np.abs((Pi - F*CR2)/Pi)), cmap='hot' )

    plt.sca(axs[0,1])
    plt.plot( Pos[1:], 'k--')
    plt.plot( Pcs[1:], 'k-')
    plt.plot( F * CR2s[1:], 'bo')
    plt.plot( Pfs[1:], 'r.')
    plt.legend(['Open','Closed','Cs','Pfs'])
    if yscale[1] == 'log':
        plt.semilogy()
    elif yscale[1] == 'loglog':
        plt.loglog()
    plt.title('Genomic distance maps profiles')
    
    plt.sca(axs[1,1])
    plt.plot( np.abs(Pfs - F * CR2s)[1:], '.')
    plt.legend(['Pfs - Cs'])
    if yscale[1] == 'log':
        plt.semilogy()
    elif yscale[1] == 'loglog':
        plt.loglog()
    plt.title('Absolute error')
    plt.savefig('plResults.png', dpi=300)



def profPlot( MM, mom=['minmax'], flag='plot'):
    
    MMs = np.zeros((MM.shape[0]))
    MMss = np.zeros((MM.shape[0]))
    
    if mom==['minmax']:
        MMsmax = np.zeros((MM.shape[0]))
        MMsmin = np.zeros((MM.shape[0]))
        for idiag in range( 0, MM.shape[0]):
            cmdiag = np.diagonal( MM, idiag)
            cmdiag = cmdiag[cmdiag.nonzero()]
            MMs[idiag] = np.nanmean(cmdiag )
            MMss[idiag] = np.nanstd(cmdiag) 
            MMsmax[idiag] = np.nanmax(cmdiag) 
            MMsmin[idiag] = np.nanmin(cmdiag) 

    elif mom[0]=='pp':
        MMsppu = np.zeros((MM.shape[0]))
        MMsppd = np.zeros((MM.shape[0]))
        for idiag in range( 0, MM.shape[0]):
            cmdiag = np.diagonal( MM, idiag)
            cmdiag = cmdiag[cmdiag.nonzero()]
            MMs[idiag] = np.nanmean(cmdiag )
            MMss[idiag] = np.nanstd(cmdiag) 
            MMsppu[idiag] = np.nanpercentile(cmdiag, mom[1]) 
            MMsppd[idiag] = np.nanpercentile(cmdiag, 100 - mom[1]) 
    
    
    if flag == 'plot':
        # plt.plot( MM[1:], 'k--')
        plt.figure()
        plt.plot( np.arange( 0, MMs.shape[0])+1, MMs, 'b')
    
        if False:
            plt.fill_between(
                np.arange( 0, MMs.shape[0])+1, 
                MMs-MMss, MMs+MMss, 
                edgecolor=None, color='r', alpha=.2, lw=0
                )
            return MMs, MMs-MMss, MMs+MMss
    
        if mom==['minmax']:
        
            plt.fill_between(
                np.arange( 0, MMs.shape[0])+1, 
                MMsmin, MMsmax, 
                edgecolor=None, color='r', alpha=.4, lw=0
                )
            return MMs, MMsmin, MMsmax
    
        elif mom[0]=='pp':
            
            plt.fill_between(
                np.arange( 0, MMs.shape[0])+1, 
                MMsppd, MMsppu, 
                edgecolor=None, color='b', alpha=.4, lw=0
                )         
            
            return MMs, MMsppd, MMsppu
    
    else:

        if False: return MMs, MMs-MMss, MMs+MMss
    
        if mom==['minmax']: return MMs, MMsmin, MMsmax
    
        elif mom[0]=='pp': return MMs, MMsppd, MMsppu        







def buildScaling( NR, alpscal, sij, strPh='interpolate'):
    
    
    PcvC = pd.read_csv( env.strData + 'globulePhase_polymerContactProbabilityProfile.csv'
                      , sep=',', index_col=None)
    
    PcvO = pd.read_csv( env.strData + 'coilPhase_polymerContactProbabilityProfile.csv'
                      , sep=',', index_col=None)
    
    


    # extend the profiles according to power law up to a scaling constant
    NRscal = int(np.ceil(alpscal * NR))
    
    sshape = PcvO.shape[0]
    if NR > sshape:
        ## extend profile to NR number of monomers if needed
        #
        gammO = -2.1
        PcvOcorrc = pd.DataFrame(
        data={
            'Pc' : np.arange( sshape, NRscal, 1.) ** gammO  * PcvO['Pc'].values[ sshape ] /  (sshape) ** gammO ,
            's' : np.arange( sshape, NRscal, 1.)
            })
        
        # join back
        PcvOcorrc = PcvO.iloc[:sshape-1].append(  PcvOcorrc )
        
        
        #     
        gammG = 0
        PcvGcorrc = pd.DataFrame(
        data={
            'Pc' : np.arange( sshape, NRscal, 1.) ** gammG  * PcvC['Pc'].values[ sshape ] /  (sshape) ** gammG ,
            's' : np.arange( sshape, NRscal, 1.)
            })    
        
        # join back
        PcvGcorrc = PcvC.iloc[:sshape-1].append(  PcvGcorrc )
    
    
    else:
        PcvOcorrc = PcvO.iloc[:NRscal-1]
        PcvGcorrc = PcvC.iloc[:NRscal-1]
            
    

    
    
    # scale the non-universal small-scale part of the curve
    
    # strPh = 'interpolate'
    if strPh == 'interpolate':
        # Interpolate
        s2 = np.arange( 1, NR+1 ) 
        
        Pcv2C = np.interp( s2, (PcvGcorrc['s']-1) / alpscal, PcvGcorrc['Pc'])
        Pcv2O = np.interp( s2, (PcvOcorrc['s']-1) / alpscal, PcvOcorrc['Pc'])        
        
        # Pcv2C = interp1d( s2, PcvGcorrc['s'], PcvGcorrc['Pc'], kind='quadratic')
        # Pcv2O = interp1d( s2, PcvO['s'], PcvO['Pc'], kind='quadratic')
    
    else:
        
        Pcv2C = PcvC.values[:NR,1].T
        Pcv2O = PcvO.values[:NR,1].T
        
    
    Pc2 = pd.DataFrame(
        data = {
            'Open' : Pcv2O ,
            'Closed' : Pcv2C
            }
        )
    
    # Pc2.to_csv( env.strStorage2 + '/SA/F_CG_'+regg+'_ext.csv', sep = ',', header=True, index=False )
    
    FF = np.copy(Pc2.values.T)
    
    ppo = FF[0,sij]
    ppc = FF[1,sij]    


    return ppo, ppc











def buildScaling2( NR, alpscal, sij, strPh='interpolate'):
    
    PcvC = pd.read_csv( env.strStorage2 + '/SA/' + 'ContactProb-fractal-globule--N512----CR-0.50CG-10.00--ER-2.0EG-0.0--.csv'
                      , sep=',', index_col=None
                      , names=['s','Pc'], header=0 )
    
    PcvO = pd.read_csv( env.strStorage2 + '/SA/' + 'ContactProb-fractal-globule--N512----CR-1.00CG-10.00--ER-2.0EG-0.0--.csv'
                      , sep=',', index_col=None
                      , names=['s','Pc'], header=0 )
    
    
    
    # extend the profiles according to power law up to a scaling constant
    NRscal = alpscal * NR
    
    
    #
    openStartBead = 120
    gammO = -2.1
    PcvOcorrc = pd.DataFrame(
    data={
        'Pc' : np.arange( openStartBead, NRscal, 1.) ** gammO  * PcvO['Pc'].values[ openStartBead ] /  (openStartBead) ** gammO ,
        's' : np.arange( openStartBead, NRscal, 1.)
        })
    
    # join back
    PcvOcorrc = PcvO.iloc[:openStartBead-1].append(  PcvOcorrc )
    
    
    #     
    globStartBead = 220
    gammG = 0
    PcvGcorrc = pd.DataFrame(
    data={
        'Pc' : np.arange( globStartBead, NRscal, 1.) ** gammG  * PcvC['Pc'].values[ globStartBead ] /  (globStartBead) ** gammG ,
        's' : np.arange( globStartBead, NRscal, 1.)
        })    
    
    # join back
    PcvGcorrc = PcvC.iloc[:globStartBead-1].append(  PcvGcorrc )
    
    
    
    
    # scale the non-universal small-scale part of the curve
    
    # strPh = 'interpolate'
    if strPh == 'interpolate':
        # Interpolate
        s2 = np.arange( 1, NR+1 ) 
        
        Pcv2C = np.interp( s2, (PcvGcorrc['s']-1) / alpscal, PcvGcorrc['Pc'])
        Pcv2O = np.interp( s2, (PcvOcorrc['s']-1) / alpscal, PcvOcorrc['Pc'])        
        
        # Pcv2C = interp1d( s2, PcvGcorrc['s'], PcvGcorrc['Pc'], kind='quadratic')
        # Pcv2O = interp1d( s2, PcvO['s'], PcvO['Pc'], kind='quadratic')
    
    else:
        
        Pcv2C = PcvC.values[:NR,1].T
        Pcv2O = PcvO.values[:NR,1].T
        
    
    Pc2 = pd.DataFrame(
        data = {
            'Open' : Pcv2O ,
            'Closed' : Pcv2C
            }
        )
    
    # Pc2.to_csv( env.strStorage2 + '/SA/F_CG_'+regg+'_ext.csv', sep = ',', header=True, index=False )
    
    FF = np.copy(Pc2.values.T)
    
    ppo = FF[0,sij]
    ppc = FF[1,sij]    


    return ppo, ppc, FF









def buildExpCM( strRegg, strStartSet = 'read+random+diag', outlimod = ['percent',99.5]):
    
    

    if strRegg == 'GSC275+5000':
        
        regg = 'GSC275+5000'
        res = '5000'
        mcoolf = 'GSC275-Arima-allReps-filtered.mcool'
        
        
        c = cooler.Cooler( env.strStorage2 + '/data/coolers/' + mcoolf,root='/resolutions/5000')
        resolution = c.binsize
        CC = c.matrix(balance=True).fetch('chr13:99,230,000-102,430,000')
    
    
    elif strRegg == 'GSC275+10000':
        
        regg = 'GSC275+10000'
        res = '10000'
        mcoolf = 'GSC275-Arima-allReps-filtered.mcool'
        
        
        c = cooler.Cooler( env.strStorage2 + '/data/coolers/' + mcoolf,root='/resolutions/10000')
        resolution = c.binsize
        CC = c.matrix(balance=True).fetch('chr13:99,230,000-105,430,000')
    
        
    
    
    
    ## set CC matrix
    CC = CC / np.nanmax(CC)
    
    # why there's 0s and nans?
    CC = np.where( np.isnan(CC), 0, CC)
    
    # strStartSet = 'read+random+diag' # 'read+random+diag' 'read+infer' 'read+random'
    if strStartSet == 'read':
        np.fill_diagonal(CC,0)
    
    elif strStartSet == 'read+infer':
        np.fill_diagonal(CC,0)
    
    
    elif strStartSet == 'read+random':
        np.fill_diagonal(CC,0)
        
        if strDiag == 'zerod':
            for ni in range( CC.shape[0]) :
                for nj in range(ni,np.minimum(ni+int(casssel.nzd),CC.shape[0])):
                    CCd[ni,nj] = 0
                    CCd[nj,ni] = 0
                    
    elif strStartSet == 'read+random+diag':
        pass
        
    
    
    
    if outlimod[0] == 'diag':
        nzd = outlimod[1]
        for ni in range(N) :
            for nj in range(ni,np.minimum(ni+nzd,N)):
                CC[ni,nj] = np.nan
                CC[nj,ni] = np.nan
                
    elif outlimod[0] == 'percent':
        CR2perc = np.percentile( CC, outlimod[1])
        CC[ CC > CR2perc] = np.nan    
    
    
    
    
    return CC










def fetchMcool( fncool, resstr, chrcoo, isbalace=True):
    
    # fetch matrix
    c = cooler.Cooler( env.strStorage2 + '/data/coolers/' + fncool,root='/resolutions/'+ resstr)
    # resolution = c.binsize
    CC = c.matrix(balance=isbalace).fetch( chrcoo )    

    return CC





def fetchMcoolSV( fncool, resstr, chrcoo, assembly, isbalace=True):
    
    # fetch cooler
    clr = cooler.Cooler( env.strStorage2 + '/data/coolers/' + fncool, root='/resolutions/' + resstr)
    # build sv
    vis = Triangle(clr, assembly, n_rows=3, figsize=(7, 4.2), track_partition=[5, 0.4, 0.5], correct='weight')

    return vis.matrix











# =============================================================================
# Plots
# =============================================================================

def plRes2( Pi, tracks, ppv=[74,99.8], cmap='YlOrRd', figsiz = (7,9)):

    M = len(tracks)+1
    fig = plt.figure(figsize=figsiz,dpi=100)
    
    widths = [1]
    # heights = [1, .02 + .005*len(tracks)]
    heights = [1, .04 + .04*len(tracks)]
    spec = fig.add_gridspec(ncols=1, nrows=2, width_ratios=widths,
                              height_ratios=heights, wspace=0.04, hspace=0)
    
    acc = fig.add_subplot(spec[0])
    atr = fig.add_subplot(spec[1], sharex=acc)
    
    
    if ppv == None:
        acc.matshow( np.log( Pi), cmap=cmap, aspect='auto')    
    else:
        acc.matshow( np.log( Pi), cmap=cmap, aspect='auto' ,
                         vmin=np.log(np.nanpercentile(Pi[Pi.nonzero()],ppv[0])), 
                         vmax=np.log(np.nanpercentile(Pi[Pi.nonzero()],ppv[1])) )    

    acc.yaxis.tick_right()
    
    
    
    # =============================================================================
    # ## plot colors distrubution
    # =============================================================================
    # colovsort = [6,4,2,3,0,1,7,5,8]

    # fig, axs = plt.subplots( 2, 1, figsize=(16,10), gridspec_kw={'hspace':0})
    ntra = tracks[0].size
    xrang = range(0, ntra)
    yref = np.ones( ntra)
    cmap = plt.cm.get_cmap('tab20', M-1)
    for mi in range(0,M-1):
        # misort = colovsort[mi]
        misort = mi
        atr.fill_between( xrang, mi + tracks[misort], mi * yref, color=cmap( (mi/(M-2)) % 1.1)
                         #, edgecolor=None
                         , linewidth=.1
                         )

        atr.set_ylim(0,M-1)
        atr.set_xlim(0, ntra)

    atr.set_yticks( np.arange(1,M)- .5 )
    atr.set_yticklabels( range(1,M) )
    # atr.axes.get_yaxis().set_visible(False)
    atr.yaxis.tick_right()


                
    
    return fig, acc, atr








def plRes3( CR2, Pi, ppv=[[20,99],[74,99.8]], cmap='YlOrRd'):
    

    fig, axs = plt.subplots(1, 2, figsize=(10,7))
        
    if ppv[0] == None:
        axs[0].matshow( np.log( CR2), cmap=cmap)    
    else:
        axs[0].matshow( np.log( CR2), cmap=cmap, 
                         vmin=np.log(np.nanpercentile(CR2[CR2.nonzero()],ppv[0][0])), 
                         vmax=np.log(np.nanpercentile(CR2[CR2.nonzero()],ppv[0][1])) )     
    
    if ppv[1] == None:
        axs[1].matshow( np.log( Pi), cmap=cmap)    
    else:
        axs[1].matshow( np.log( Pi), cmap=cmap, 
                         vmin=np.log(np.nanpercentile(Pi[Pi.nonzero()],ppv[1][0])), 
                         vmax=np.log(np.nanpercentile(Pi[Pi.nonzero()],ppv[1][1])) ) 





def plRes3b( CR2, Pi, ppv=[[20,99],[74,99.8]], cmap='YlOrRd', cmnorm = 'log'):
    
    v0min=np.nanpercentile(CR2[CR2.nonzero()],ppv[0][0])    
    v0max=np.nanpercentile(CR2[CR2.nonzero()],ppv[0][1])    
    v1min=np.nanpercentile(CR2[CR2.nonzero()],ppv[1][0])    
    v1max=np.nanpercentile(CR2[CR2.nonzero()],ppv[1][1])    

    #
    countmin0, countmax0, countstep0 = v0min, v0max, ( v0max - v0min) / 100
    countmin1, countmax1, countstep1 = v1min, v1max, ( v1max - v1min) / 100
    countvec0 = np.arange( countmin0, countmax0, countstep0)
    countvec1 = np.arange( countmin1, countmax1, countstep1)

    # pdb.set_trace()
    # build cm scaling        
    if cmnorm == cmnorm:
        colnorm0 = mpl.colors.LogNorm(vmin=countmin0, vmax=countmax0)
        colnorm1 = mpl.colors.LogNorm(vmin=countmin1, vmax=countmax1)
    elif cmnorm == cmnorm:
        colnorm0 = mpl.colors.Normalize(vmin=countmin0, vmax=countmax0)
        colnorm1 = mpl.colors.Normalize(vmin=countmin1, vmax=countmax1)
    else:
        colnorm0 = mpl.colors.LogNorm(vmin=countmin0, vmax=countmax0)
        colnorm1 = mpl.colors.LogNorm(vmin=countmin1, vmax=countmax1)
        
    # check if cm is given as string
    if type( cmap) is not list:
        if cmap in plt.colormaps():
            cm0 = cmap
            cm1 = cmap
        else:
            cmv = [ chi for chi in cmap]
            #
            cm0 = mpl.colors.LinearSegmentedColormap.from_list('HiCcount',    cmv , N=len(countvec0) -1)
            cm1 = mpl.colors.LinearSegmentedColormap.from_list('HiCcount',    cmv , N=len(countvec1) -1)
        
    else :
        cmv = cmap
        #
        cm0 = mpl.colors.LinearSegmentedColormap.from_list('HiCcount',    cmv , N=len(countvec0) -1)    
        cm1 = mpl.colors.LinearSegmentedColormap.from_list('HiCcount',    cmv , N=len(countvec1) -1)    
    
    

    fig, axs = plt.subplots(1, 2, figsize=(12,6))
        
    if ppv[0] == None:
        axs[0].matshow( np.log( CR2), cmap=cm4)    
    else:
        axs[0].matshow( CR2, 
                        interpolation='none', 
                        cmap= cm0, 
                        norm =  colnorm0,
                        # vmin= countvec[0], vmax= countvec[-1] ,
                        aspect='auto', 
                        # vmin=np.log(np.nanpercentile(CR2[CR2.nonzero()],ppv[0][0])), 
                        # vmax=np.log(np.nanpercentile(CR2[CR2.nonzero()],ppv[0][1])) 
                         )     
    
    
    if ppv[1] == None:
        axs[1].matshow( np.log( Pi), cmap=cm4)    
    else:
        axs[1].matshow( Pi, 
                        interpolation='none', 
                        cmap= cm1, 
                        norm =  colnorm1,
                        # vmin= countvec[0], vmax= countvec[-1] ,
                        aspect='auto', 
                        # vmin=np.log(np.nanpercentile(Pi[Pi.nonzero()],ppv[1][0])), 
                        # vmax=np.log(np.nanpercentile(Pi[Pi.nonzero()],ppv[1][1])) 
                        ) 









def plRes5( CR2, Pi, ppv=[[20,99],[74,99.8]], ext=None, cmap='YlOrRd'):

    # fig, axs = plt.subplots(1, 2, figsize=(10,7))
        
    fig = plt.figure(figsize=(10,7),dpi=100)
    
    widths = [1, 1]
    # heights = [1, .02 + .005*len(tracks)]
    heights = [1]
    spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=widths,
                              height_ratios=heights, wspace=0.04, hspace=0)
    
    acc = fig.add_subplot(spec[ 0])
    as1 = fig.add_subplot(spec[ 1], sharex=acc, sharey=acc)
   
    
    if ppv[0] == None:
        acc.matshow( np.log( CR2), cmap=cmap, extent=ext)    
    else:
        acc.matshow( np.log( CR2), cmap=cmap, 
                         vmin=np.log(np.nanpercentile(CR2[CR2.nonzero()],ppv[0][0])), 
                         vmax=np.log(np.nanpercentile(CR2[CR2.nonzero()],ppv[0][1])) , extent=ext)     
    
    if ppv[1] == None:
        as1.matshow( np.log( Pi), cmap=cmap, extent=ext)    
    else:
        as1.matshow( np.log( Pi), cmap=cmap, 
                         vmin=np.log(np.nanpercentile(Pi[Pi.nonzero()],ppv[1][0])), 
                         vmax=np.log(np.nanpercentile(Pi[Pi.nonzero()],ppv[1][1])) , extent=ext) 

    as1.yaxis.tick_right()

    return acc, as1, fig





def plRes4( Pi, CR2, ttcsel, R, M, ppv=[[20,99],[74,99.8]], cmap='YlOrRd'):

    
    Ri = np.arange( 0, ttcsel.size, R, dtype=np.int_)

    tracks = []
    for mi in range(1, M):
        tracks += [np.add.reduceat( np.int_(ttcsel == mi), Ri)/R]
                

    fig = plt.figure(figsize=(11,9),dpi=100)
    
    widths = [1, 1]
    # heights = [1, .02 + .005*len(tracks)]
    heights = [1, .04 + .04*len(tracks)]
    spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
                              height_ratios=heights, wspace=0.04, hspace=0)
    
    acc = fig.add_subplot(spec[0, 0])
    as1 = fig.add_subplot(spec[0, 1], sharex=acc, sharey=acc)
    atr = fig.add_subplot(spec[1, 1], sharex=as1)
    
    
    if ppv[0] == None:
        acc.matshow( np.log( Pi), cmap=cmap, aspect='auto')    
    else:
        acc.matshow( np.log( Pi), cmap=cmap, aspect='auto' ,
                         vmin=np.log(np.nanpercentile(Pi[Pi.nonzero()],ppv[0][0])), 
                         vmax=np.log(np.nanpercentile(Pi[Pi.nonzero()],ppv[0][1])) )    
    if ppv[1] == None:
        as1.matshow( np.log( CR2), cmap=cmap, aspect='auto' )
    else:
        as1.matshow( np.log( CR2), cmap=cmap, aspect='auto' ,
                         vmin=np.log(np.nanpercentile(CR2[CR2.nonzero()],ppv[1][0])), 
                         vmax=np.log(np.nanpercentile(CR2[CR2.nonzero()],ppv[1][1])) )    

    as1.yaxis.tick_right()
    
    
    
    # =============================================================================
    # ## plot colors distrubution
    # =============================================================================
    # colovsort = [6,4,2,3,0,1,7,5,8]

    # fig, axs = plt.subplots( 2, 1, figsize=(16,10), gridspec_kw={'hspace':0})
    ntra = tracks[0].size
    xrang = range(0, ntra)
    yref = np.ones( ntra)
    cmap = plt.cm.get_cmap('tab20', M-1)
    for mi in range(0,M-1):
        # misort = colovsort[mi]
        misort = mi
        atr.fill_between( xrang, mi + tracks[misort], mi * yref, color=cmap( (mi/(M-2)) % 1.1)
                         #, edgecolor=None
                         , linewidth=.1
                         )

        atr.set_ylim(0,M-1)
        atr.set_xlim(0, ntra)

    atr.set_yticks( np.arange(1,M)- .5 )
    atr.set_yticklabels( range(1,M) )
    # atr.axes.get_yaxis().set_visible(False)
    atr.yaxis.tick_right()


    if False:
        nav = 80
        colov = []
        for mi in range(1, M):
            colov += [
                np.convolve( np.int_(ttcsel == mi), np.ones(nav)/nav)
                ]
                
    
        ntra = tracks[0].size
    
        fig, axs = plt.subplots( 2, 1, figsize=(16,10), gridspec_kw={'hspace':0})
        cmap = plt.cm.get_cmap('tab20', M-1)
        for mi in range(0,M-1):
            # misort = colovsort[mi]
            misort = mi
            axs[1].plot( colov[misort], color=cmap( (mi/(M-2)) % 1.1), alpha=.7)

            axs[1].set_ylim(0,1)
            axs[1].set_xlim(0,ntra)
            axs[1].axes.get_yaxis().set_visible(False)
                
    
    return fig, acc, as1, atr






def plRes6( Pi, Pi2, ttcsel, ttcsel2, R, M, ppv=[[20,99],[74,99.8]], cmap='YlOrRd'):

    
    Ri = np.arange( 0, ttcsel.size, R, dtype=np.int_)

    tracks = []
    for mi in range(1, M):
        tracks += [np.add.reduceat( np.int_(ttcsel == mi), Ri)/R]
        
    tracks2 = []
    for mi in range(1, M):
        tracks2 += [np.add.reduceat( np.int_(ttcsel2 == mi), Ri)/R]        
                

    fig = plt.figure(figsize=(11,9),dpi=100)
    
    widths = [1, 1]
    # heights = [1, .02 + .005*len(tracks)]
    heights = [1, .04 + .04*len(tracks)]
    spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
                              height_ratios=heights, wspace=0.04, hspace=0)
    
    acc = fig.add_subplot(spec[0, 0])
    as1 = fig.add_subplot(spec[0, 1], sharex=acc, sharey=acc)
    atr1 = fig.add_subplot(spec[1, 0], sharex=as1)
    atr2 = fig.add_subplot(spec[1, 1], sharex=as1, sharey=atr1)
    
    
    if ppv[0] == None:
        acc.matshow( np.log( Pi), cmap=cmap, aspect='auto')    
    else:
        acc.matshow( np.log( Pi), cmap=cmap, aspect='auto' ,
                         vmin=np.log(np.nanpercentile(Pi[Pi.nonzero()],ppv[0][0])), 
                         vmax=np.log(np.nanpercentile(Pi[Pi.nonzero()],ppv[0][1])) )    
    if ppv[1] == None:
        as1.matshow( np.log( Pi2), cmap=cmap, aspect='auto' )
    else:
        as1.matshow( np.log( Pi2), cmap=cmap, aspect='auto' ,
                         vmin=np.log(np.nanpercentile(Pi2[Pi2.nonzero()],ppv[1][0])), 
                         vmax=np.log(np.nanpercentile(Pi2[Pi2.nonzero()],ppv[1][1])) )    

    as1.yaxis.tick_right()
    
    
    
    # =============================================================================
    # ## plot colors distrubution
    # =============================================================================
    # colovsort = [6,4,2,3,0,1,7,5,8]

    # fig, axs = plt.subplots( 2, 1, figsize=(16,10), gridspec_kw={'hspace':0})
    ntra = tracks[0].size
    xrang = range(0, ntra)
    yref = np.ones( ntra)
    cmap = plt.cm.get_cmap('tab20', M-1)
    for mi in range(0,M-1):
        # misort = colovsort[mi]
        misort = mi
        atr1.fill_between( xrang, mi + tracks[misort], mi * yref, color=cmap( (mi/(M-2)) % 1.1))

        atr1.set_ylim(0,M-1)
        atr1.set_xlim(0, ntra)

    atr1.set_yticks( np.arange(1,M)- .5 )
    atr1.set_yticklabels( range(1,M) )
    # atr1.axes.get_yaxis().set_visible(False)
    # atr1.yaxis.tick_right()


    ### ttc mod
    ntra = tracks2[0].size
    xrang = range(0, ntra)
    yref = np.ones( ntra)
    cmap = plt.cm.get_cmap('tab20', M-1)
    for mi in range(0,M-1):
        # misort = colovsort[mi]
        misort = mi
        atr2.fill_between( xrang, mi + tracks2[misort], mi * yref, color=cmap( (mi/(M-2)) % 1.1))

        atr2.set_ylim(0,M-1)
        atr2.set_xlim(0, ntra)

    atr2.set_yticks( np.arange(1,M)- .5 )
    atr2.set_yticklabels( range(1,M) )
    # atr2.axes.get_yaxis().set_visible(False)
    atr2.yaxis.tick_right()






















