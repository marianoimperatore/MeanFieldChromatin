#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 18:06:45 2023

@author: mariano
"""




import sys, os
import cooler
import matplotlib.pyplot as plt
import numpy as np
import scipy as spy
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import pandas as pd
from numba import njit, vectorize
import random
import pdb
import re
from itertools import product
import importlib
import warnings
warnings.filterwarnings("ignore")

#### import simulation settings
if False:
    os.chdir( '/usr/users/mariano/inverse' )
    
    

import esprenv as env
import funinv as func0
import setInvDict as invdict




if False:
    strcmd = '$HOME/inverse/infer_bs.py nod:1,30,40,4,12345 reg:GSC275B,C88bex2,5000,missingRegions param:6 cfmax:.8,1 pct:99.6,99.9 filt:.5'
    sys.argv = re.split( ' +', strcmd)
    



# =============================================================================
# # read input settings
# =============================================================================


for argvl in sys.argv[1:]:
    
    argvls = argvl.split(':')
    argvls1 = argvls[0]
    argvls2 = argvls[1].split(',')
    
    if argvls1=='reg':
        
        regg = '+'.join( [
            argvls2[0] ,  argvls2[1] ,  argvls2[2]
            ] )
        fncool = invdict.coolDict[ argvls2[0]][0]
        chrcoo = invdict.coolDict[ argvls2[0]][1][ argvls2[1]][ argvls2[2]][0]

        resstr = argvls2[2]
        res = int(argvls2[2])

        strInputType = argvls2[3]
        Cnanflag = 'nanlow'
        Clowpp = 5

        continue
    
    if argvls1=='param':
        M = int(argvls2[0])
        continue    

    if argvls1=='cfmax':
        inpv = [ float(inpi) for inpi in argvls2]  
        strInputM = 'cfmax'
        continue
    
    if argvls1=='pct':
        pctv = [ float(pcti) for pcti in argvls2]
        continue 

    if argvls1=='filt':
        strFilt = True
        sigv = [ float(sigi) for sigi in argvls2]
        continue 
    else:
        strFilt= False
        sigv = [np.nan]

    
    if argvls1=='nod':
        nodid, nnod, npernod, nens, rseed = [int(nodi) for nodi in argvls2]
        continue            
        





# fetch matrix
try:
    c = cooler.Cooler( env.strStorEspr + fncool,root='/resolutions/'+ resstr)
    # resolution = c.binsize
    CC = c.matrix(balance=True).fetch( chrcoo )
    
    
except:    
    CC = pd.read_csv( re.sub('.mcool','-',fncool)+ regg +'.csv', index_col=False).values





# =============================================================================
# Initiate and process contact profiles of thermodynamic phases
# =============================================================================
# settings
N = CC.shape[0]
R = np.copy(M)
NR = N * R
N2R4 = N ** 2 * R**4
R2 = R**2
N2 = N**2


Ri = np.arange( 0, NR, R, dtype=np.int_)
Rir = np.arange( 0, NR, R, dtype=np.int_)
Rij = Ri[None,:] - Ri[:,None]




## sij per NR 
diaggi = np.diag_indices( NR, ndim=2) 
sij = np.zeros((NR,NR), dtype=np.int_)

for digi in range(0,NR):
    sij[ (diaggi[0]+digi) % NR, diaggi[1] ] = digi

sij = np.tril( sij )+ np.tril(sij).T

## sij per N
diaggiN = np.diag_indices( N, ndim=2) 
sijN = np.zeros((N,N), dtype=np.int_)

for digi in range(0,N):
    sijN[ (diaggiN[0]+digi) % N, diaggiN[1] ] = digi

sijN = np.tril( sijN )+ np.tril(sijN).T







# =============================================================================
# ### process input simul matrix
# =============================================================================

PcvC = pd.read_csv( env.strStorEspr + 'globulePhase_polymerContactProbabilityProfile.csv'
                  , sep=',', index_col=None)

PcvO = pd.read_csv( env.strStorEspr + 'coilPhase_polymerContactProbabilityProfile.csv'
                  , sep=',', index_col=None)









# =============================================================================
# Select parameters range for grid search
# =============================================================================
alpscalv = np.arange( .8,1.3,.1) 
alpv = np.arange( .3,.8,.1) 
Fv = np.arange(.55,1.45,.15)
bettv = np.exp( np.log(2) * np.arange( 6.6,9.2,.5) ) 
lambv = np.exp( np.log(2) * np.arange( -9,-4,.5) ) 
cfstr = ''
strParams = ''




    

### grid search dimensions
nsiz = len(alpscalv) * len(inpv) * len(alpv) * len(Fv) * len(bettv) * len(lambv) * len(sigv) * nens
gridsltot = list( product( alpscalv, inpv, alpv, Fv, bettv, lambv, sigv) )
nnod * npernod * nens


# random sample
random.seed(rseed)
rsampl = random.sample( gridsltot, npernod * nnod )

# select node's grid-points
gridsl = rsampl[ (nodid-1) * npernod : nodid * npernod]




##
resultspd = pd.DataFrame( [])
nsavi = 0
nsave = 20










# =============================================================================
# Start grid search
# =============================================================================
for gridsli in gridsl:
    # gridsli = gridsl[0]
    alpscal = gridsli[ 0 ]
    
    alp = gridsli[ 2 ]
    F = gridsli[ 3 ]
    beta1 = gridsli[ 4 ]
    lamEntrop0 = gridsli[ 5 ]
    sig = gridsli[ 6 ]



    
    # =============================================================================
    # # modify input matrix
    # =============================================================================

    if (strFilt) and (sig!=0):
        CCf = gaussian_filter( CC, sigma=sig )
        
        crfilte = np.isnan(CCf) & ~np.isnan(CC)
        CCf[ crfilte] = CC[ crfilte]        
        
        # list of beads to exclude because of nan
        nanrow = np.isnan(CC).all(0)
        nanmatremove = ~ ( nanrow[:,None] | nanrow[None,:] )
        removebbN = np.where( nanrow)[0]
        rmaux = np.array(list(range(0,M)) * removebbN.size)
        removebb = np.repeat( removebbN * M, M) + rmaux
        
    else:
        CCf = np.copy( CC)
        removebb = np.array([])
        nanmatremove = np.ones(CC.shape, dtype=bool)




    ####### substitute zero or low/high percentile to isolated nans
    if Cnanflag=='nanlow':
        mom = ['pp',Clowpp] 
        profi = func0.profPlot( CCf, mom=mom, flag='noplot')
        
        # subst percentile of CC distribution per given distance
        CCf[np.isnan( CCf) & nanmatremove] = profi[2][ sijN[ np.isnan( CCf) & nanmatremove] ]
        
        # still nans? put them to zero
        CCf = np.where( np.isnan(CCf) & nanmatremove, 0, CCf)    
    
    
    elif Cnanflag=='nanhigh':
        mom = ['pp',Chighpp] 
        profi = func0.profPlot( CCf, mom=mom, flag='noplot')
        
        # subst percentile of CCf distribution per given distance
        CCf[np.isnan( CCf) & nanmatremove] = profi[1][ sijN[ np.isnan( CCf) & nanmatremove] ]
        
        # still nans? put them to zero
        CCf = np.where( np.isnan(CCf) & nanmatremove, 0, CCf)    
    
    else:
        CCf = np.where( np.isnan(CCf) & nanmatremove, 0, CCf)    
    
    
    








    ####
    nzd = 1
    cfmax = gridsli[ 1 ]
    
    # 
    CCcorrl = []
    for idpct, pcti in enumerate(pctv):
        CCcorr = np.copy( CCf) * R2
        # why there's 0s and nans?
        # CCcorr = np.where( np.isnan(CCcorr), 0, CCcorr)
        
        CCpct = np.percentile( CCcorr, pcti)
        CCcorr[ CCcorr > CCpct] = np.nan
        CCcorrl += [CCcorr]

    # 
    CCd = np.copy( CCf)
    
    # why there's 0s and nans?
    # CCd = np.where( np.isnan(CCd), 0, CCd)
    
    # ceil at 10% of max
    CCd[CCd > CCd.max() * cfmax] = CCd.max() * cfmax
    
    # standardize
    CCd = CCd / np.nanmax(CCd)
    
    # 
    # zeroing diagonal 
    np.fill_diagonal(CCd,0)
    


    
            
    
    
    
    # extend the profiles according to power law up to a scaling constant
    NRscal = alpscal * NR
    
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
    strPh = 'interpolate'
    if strPh == 'interpolate':
        # Interpolate
        s2 = np.arange( 1, NR+1 ) 
        
        Pcv2C = np.interp( s2, (PcvGcorrc['s']-1)/alpscal, PcvGcorrc['Pc'])
        Pcv2O = np.interp( s2, (PcvOcorrc['s']-1)/alpscal, PcvOcorrc['Pc'])        
    
    else:
        Pcv2C = PcvC.values[:NR,1].T
        Pcv2O = PcvO.values[:NR,1].T
        
    
    Pc2 = pd.DataFrame(
        data = {
            'Open' : Pcv2O ,
            'Closed' : Pcv2C
            }
        )
    
    FF = np.copy(Pc2.values.T)
    
    
    
    
    ppos1 = np.add.reduceat( FF[0,:], Rir, 0) 
    ppcs1 = np.add.reduceat( FF[1,:], Rir, 0)
    
    
    
    ### put to zero the diagonal term of the polymer profile    
    FF[:,: R * nzd] = 0
    
    ### 
    ppo = FF[0,sij]
    ppc = FF[1,sij]
    
    
    ppos = np.add.reduceat( FF[0,:], Rir, 0) 
    ppcs = np.add.reduceat( FF[1,:], Rir, 0)
    
    
    

        
        
        
        
        
        


    for ensi in range( nens ):
        # ensi = 0
                
        
        
        
        # 
        ttcc = np.random.randint(0,M,NR, dtype=np.int_)
        ttco = np.zeros((NR))
        
        
        if strInputType == 'missingRegions':
            ttcc[ removebb] = 0
        
        

        
        
        # =============================================================================
        # Functions
        # =============================================================================
        
        def buildPnodiag3( bb):
            
            ppi = np.where( (bb[None,:] == bb[:, None]) & (bb[None,:] != 0) & (bb[:,None] != 0), ppc, ppo )
            ppi = np.where( (bb[None,:] != bb[:, None]) & (bb[None,:] != 0) & (bb[:,None] != 0), 0, ppi )
            
            ppir = np.add.reduceat( np.add.reduceat( ppi, Rir), Rir, 1)    
        
            np.fill_diagonal( ppir, 0)
            
            return ppir        
        
    
        
        
        
        
        
        def calcDPeff2( x1, xx1, bb):
            ppf = np.where( (bb[None,:] == bb[ x1, None]) & (bb[None,:] != 0) & (bb[ x1, None] != 0), ppc[ x1,:], ppo[ x1,:] )
            ppi = np.where( (ttc[None,:] == ttc[ x1, None]) & (ttc[None,:] != 0) & (ttc[ x1, None] != 0), ppc[ x1,:], ppo[ x1,:] )
        
            ppf = np.where( (bb[None,:] == bb[ x1, None]) & (bb[None,:] != 0) & (bb[ x1, None] != 0), ppc[ x1,:], ppo[ x1,:] )
            ppi = np.where( (ttc[None,:] == ttc[ x1, None]) & (ttc[None,:] != 0) & (ttc[ x1, None] != 0), ppc[ x1,:], ppo[ x1,:] )
        
            dpp = np.add.reduceat( ppf, Rir, 1) - np.add.reduceat( ppi, Rir, 1)  
            dpp[:,xx1] = 0
            
            return dpp.flatten() # - Pi[xx1,:]
        
        
        
        
        
        
        def CostFeff( Pi, alp, F, CR2):
            
            H = np.sqrt( np.nansum( (alp * Pi + (1-alp) * Po - F * CR2)**2  ) / N2 )
        
            return H
        


        
        def DcEaF3( DP, xx1):
                
            return np.nansum( alp2 * 2*(DP**2) +  2* DP * aFPoPC[xx1])
        
        
        
        
        

        
        
        
        
        
        
        # =============================================================================
        # Read stuff
        # =============================================================================
        
        #
        ttco = np.zeros((NR))
        Po = buildPnodiag3( ttco)

        
        # =============================================================================
        # Settings
        # =============================================================================
        strTransf = 'no'
        
        
        
        
        ##
        ttc = np.copy(ttcc)
        
        #
        CR2= CCd*R2
        sC = CCd.sum()
        sCCR2 = np.nansum(CR2)
        Po2 = (Po**2).sum()
        sPo = Po.sum()
        sCR2 = np.nansum(CR2**2)
        sCR2Po = np.nansum(CR2 * Po)
        #
        Pi = buildPnodiag3( ttc)
        sCR2P = np.nansum(CR2 * Pi)
        sPPo = (Pi*Po).sum()
        sP2 = (Pi**2).sum()
        
        # Entropy
        Sf = np.arange( 0,NR+1,1)/NR * np.log( np.arange( 0,NR+1,1)/NR)
        uni = np.unique(np.int_(ttc), return_counts=True)
        ncou = np.zeros((M), dtype=np.int_)
        ncou[ uni[ 0]] = uni[ 1]
        si = np.nansum(Sf[ncou])
        
        
        # Entropy 2
        uni2 = np.unique( list(ttc)+list(range(0,M)), return_counts=True)
        ncou2 = uni2[1]-1
        nint = ncou2[1:].sum()
        
        
        
        
        # =============================================================================
        # Settings
        # =============================================================================
        ##
        eplen = 5
        eplen3 = 50
        
        ###
        q1 = 1-.001*eplen3; q2 = .001*eplen3;
        
        bbet = beta1 * (np.log(q2)/np.log(q1)) ** (np.arange(0,eplen,1)/(eplen-1)) # / NR
        lamEntrop = lamEntrop0 * 1
        
        
        ###
        accrateprec = 400
        taui = 1
        NRprop = accrateprec * taui
        epv = [np.int_(NRprop)] * eplen
        epv[0] = np.int_(accrateprec)
        epv[-1] = np.int_(5 * accrateprec)
        
        
        ### 
        epvlen2 = 10
        epvv = range(epvlen2)
        
        errcthr = 4000
        
        
        

        
        
        ### list of class displacements available for random sampling
        rangedc = list(range(-M+1,M))
        rangedc.remove(0)
        
        ### list of bs available for random sampling
        bslist = list(range(0,NR,1))
        if strInputType == 'missingRegions':
            bslist = [ bsli for bsli in bslist if bsli not in removebb]
        
        
        
        
        
        ###
        # store results
        resutmp = pd.DataFrame( {
                    'N':[N], 'M':[M], 'R':[R],
                    'sig':[sig],
                    'ep1':[np.nan], 'ep2':[np.nan], 'ep3len':[np.nan], 'ep4len':[eplen3], 
                    'beta1':[beta1], 'q1':[q1], 'q2':q2, 'lambdaS':[lamEntrop], 'errcthr':[errcthr],
                    'cost':[np.nan], 'F':[F], 'alp':[alp], 'alpscal':[alpscal],
                    'acc':[np.nan], 'aN':[np.nan], 'aP':[np.nan], 'aR':[np.nan],
                    'aS':[np.nan], 'ncou': str(ncou2), 'cfmax':[cfmax]
                    } )

            
            
        resutmp = pd.concat( [resutmp, pd.DataFrame(ttc).T.astype(int)],axis=1)
        resultspd = resultspd.append( resutmp)
        
        
        
        
        ####
        alp2 = alp ** 2
        alpm1 = 1-alp
        alp1malp = alp*(1-alp)
        aFPoC = 2 * alp*(1-alp) * Po - 2 * alp * F * CR2
        aFPoPC = 2 * alp2 * Pi + aFPoC
        
        
        
        ##
        errc = 0 
        ##
        for epi1 in epvv :
            # epi1 = epvv[0]
            
            ##
            for idepi, epi2 in enumerate(epv) :
                
                ##
                costDHv = np.zeros(eplen3)
                
                # KBT
                rthr = - np.log( np.random.uniform(size=epi2)) 
                
           
                ##
                accp = 0; accn = 0; accr = 0
                for idep3, epi3 in enumerate(range( epi2)):
                
         
                
                    ### random new sites
                    xsitet = np.array(random.sample( bslist, eplen3)) # np.random.randint(0,NR,100)
                    xx1v = np.int_(np.floor(xsitet/R))

                    dc = np.random.choice( rangedc,size=eplen3)
                    # 
                    tcfV = [np.copy(ttc) for i in range( xsitet.size)]
            

                    # Regularization term 
                    sf = np.int_((ttc[xsitet] + dc) % (M) != 0) - np.int_(ttc[xsitet] != 0)
                    
                    
                    
                    ##
                    for idtc, tcf in enumerate(tcfV):
                        tcf[xsitet[idtc]] = (tcf[ xsitet[idtc]] + dc[idtc]) % (M)

                        ##########
                        DP = calcDPeff2( xsitet[idtc], xx1v[idtc],tcf)
                    
                        costDHv[idtc] = DcEaF3( DP, xx1v[idtc])
                        
                        
                    ## free energy 
                    costFree = costDHv / NR + lamEntrop * sf
                    
                    ## minimize
                    argmincost = np.argmin( costFree) 
                    
                    
                    ## acceptance 
                    if costFree[argmincost] <= 0:
                        ## sub new values
                        ttcdc = (ttc[xsitet[argmincost]] + dc[argmincost]) % (M)
                        

                        #
                        xx1 = xx1v[argmincost]
                        DP = calcDPeff2( xsitet[argmincost], xx1, tcfV[argmincost])
                        Pi[xx1,:] += DP
                        Pi[:,xx1] += DP
                        ttc[xsitet[argmincost]] = ttcdc
        
                        ##
                        aFPoPC = 2 * alp2 * Pi + aFPoC
        
                                        
                        ##
                        accn +=1
            
                    elif costFree[argmincost] * bbet[idepi] < rthr[idep3] :
                        ## sub new values
                        ttcdc = (ttc[xsitet[argmincost]] + dc[argmincost]) % (M)
                        #
                        xx1 = xx1v[argmincost]
                        DP = calcDPeff2( xsitet[argmincost], xx1,tcfV[argmincost])
                        Pi[xx1,:] += DP
                        Pi[:,xx1] += DP
                        ttc[xsitet[argmincost]] = ttcdc
        
                        ##
                        aFPoPC = 2 * alp2 * Pi + aFPoC
        
                        
                        ##
                        accp +=1
            
                    else:
                        accr +=1
                        continue
                    
                    
                    
                    # check error propagation
                    if errc == errcthr:
                        errc=0
                        # balance for error propagation
                        #
                        Perr = np.copy(Pi)
                        Pi = buildPnodiag3( ttc)
                        idxpi = Pi.nonzero()
                        pperr = np.nanmean(np.abs((Perr[idxpi]-Pi[idxpi])/Pi[idxpi]))
                        print('pperr:','%.3e' % pperr)
                    else:
                        errc+=1
                        
                
                
                
                Pi = buildPnodiag3( ttc)
            
                # num of beads per color
                uni2 = np.unique( list(ttc)+list(range(0,M)), return_counts=True)
                ncou2 = uni2[1]-1        
                
    
                
                Hf = CostFeff( Pi, alp, F, CR2) / R2
                
                # store results
                resutmp = pd.DataFrame( {
                            'N':[N], 'M':[M], 'R':[R],'sig':[sig],
                            'ep1':[epi1], 'ep2':[idepi], 'ep3len':[epi2], 'ep4len':[eplen3], 
                            'beta1':[beta1], 'q1':[q1], 'q2':q2, 'lambdaS':[lamEntrop], 'errcthr':[errcthr],
                            'cost':[Hf], 'F':[F], 'alp':[alp], 'alpscal':[alpscal], 
                            'acc':[(accn+accp)/epi2], 'aN':[accn/epi2], 'aP':[accp/epi2], 'aR':[accr/epi2],
                            'aS':[accp/epi2/(epi2-accn)], 'ncou': str(ncou2), 'cfmax':[cfmax]
                            } )
    
                
                
                resutmp = pd.concat( [resutmp, pd.DataFrame(ttc).T.astype(int)], axis=1)
                resultspd = resultspd.append( resutmp)        
    
    
            
                print( 'Ep:', len(epv)-(idepi+1), 'Hrel:','%.3e' % Hf, 'F:','%.2f' % (F),
                      'alp:','%.2f' % (alp), 'scal:','%.2f' % (alpscal), 'lmb:', '%.1e' % (lamEntrop), 
                      'aN:','%.1e' %(accn/epi2), 'aP:','%.1e' %(accp/epi2),'aR:','%.1e' %(accr/epi2),
                      'aS:','%.1e' % (accp/epi2/(epi2-accn)),
                      'ncou:',ncou2, 
                      )    
        
    
        
        
        

        # =============================================================================
        # Save or not to save    
        # =============================================================================
        if nsavi == nsave:
            nsavi = 0
            resultspd.to_csv( env.strStorEspr + 'resultspd_'+regg+'_N'+str(N)+'_M'+str(M)+'_id'+str(nodid)+'.csv.gz'
                             , sep = ',', header=True, index=False, compression='gzip' )
    
        else :
            nsavi +=1



# =============================================================================
# Final save
# =============================================================================
resultspd.to_csv( env.strStorEspr + 'resultspd_'+regg+'_N'+str(N)+'_M'+str(M)+'_id'+str(nodid)+'.csv.gz'
                 , sep = ',', header=True, index=False, compression='gzip' )

    




