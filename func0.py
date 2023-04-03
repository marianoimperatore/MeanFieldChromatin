#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 17:27:44 2021

@author: mariano
"""

import random
import numpy as np
import scipy as scy
from scipy.spatial import distance_matrix
import scipy.stats
import scipy.signal as scygnal
import pandas as pd
import re
import os, sys
import importlib
import itertools
import pdb
import time 

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
mpl.__version__

import linecache


#### import simulation settings
import esprenv 
import esprenv as env
from esprenv import *

if hasattr(__builtins__, '__IPYTHON__'):
    importlib.reload( esprenv )
    from esprenv import *



    
    

# set some warnings
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None  # default='warn'


def argvRead( Argv):
    # check if user edits some params
    argl = Argv[1:]
    argvtup = [ (idx,cosi)  for idx, cosi in enumerate(argl) if re.search( '-', cosi)]
    
    diddi = {}
    for argi in argvtup:
        if argi[1] == '-mint':
            diddi.update({'mint': np.int_(argl[ argi[0]+1] )})
        elif argi[1] == '-maxt':
            diddi.update({'maxt': np.int_(argl[ argi[0]+1] )})
        elif argi[1] == '-tit':
            diddi.update({'tit': np.int_(argl[ argi[0]+1] )})
        elif argi[1] == '-procid':
            diddi.update({'procid': argl[ argi[0]+1] })
        elif argi[1] == '-sysname':
            diddi.update({'str_sys': argl[ argi[0]+1] })
        elif argi[1] == '-tsam':
            diddi.update({'str_tsam': argl[ argi[0]+1] })
        elif argi[1] == '-model':
            diddi.update({'str_syst': argl[ argi[0]+1] })            
        elif argi[1] == '-param':
            diddi.update({'str_param': argl[ argi[0]+1] })            
        elif argi[1] == '-system':
            diddi.update({'strSetFile': argl[ argi[0]+1] })            
        elif argi[1] == '-mode':
            diddi.update({'str_mode': argl[ argi[0]+1] })            
        elif argi[1] == '-nMPI':
            diddi.update({'nMPI': np.int_(argl[ argi[0]+1] ) })   
            print('MPI run. no. of nodes:' , diddi['nMPI'])
        elif argi[1] == '-maxt2':
            diddi.update({'maxt2': np.int_(argl[ argi[0]+1] )})
        elif argi[1] == '-dyn1':
            diddi.update({'dyn1': argl[ argi[0]+1] })
        elif argi[1] == '-dyn2':
            diddi.update({'dyn2': argl[ argi[0]+1] })
        elif argi[1] == '-modelName':
            diddi.update({'modelName': argl[ argi[0]+1] })    
        elif argi[1] == '-SAopt':
            diddi.update({'SAopt': argl[ argi[0]+1] })     
        elif argi[1] == '-scaleback':   
            diddi.update({'scaleback': True })                 
        elif argi[1] == '-svfile':
            diddi.update({'svfile': argl[ argi[0]+1] }) 
        elif argi[1] == '-therm':
            # diddi.update({'strTherm': [argl[ argi[0]+1]] })   
            diddi.update({'strTherm': argl[ argi[0]+1].split(',') })   
        elif argi[1] == '-c':
            diddi.update({'strC': argl[ argi[0]+1] })
        elif argi[1] == '-dopt':
            diddi.update({'otheropts': argl[ argi[0]+1] })
        elif argi[1] == '-reg':
            diddi.update({'strReg': argl[ argi[0]+1] })
        else:
            print('option',argi[1],'not recognized')

    return diddi





def argvRead2( Argv, localKeys):
    # check if user edits some params
    argl = Argv[1:]
    argvtup = [ (idx, re.sub('-','', cosi) )  for idx, cosi in enumerate(argl) if re.search( '-', cosi)]
    
    diddi = {}
    for argi in argvtup:
        diddi.update({ argi[1]: argl[ argi[0]+1] })
        if argi[1] not in localKeys:
            print( 'updated variable ', argi[1])
        else:
            print( 'created variable ', argi[1], 'no previously assigned')

    return diddi












# =============================================================================
# Check if user edits some params
# =============================================================================
locals().update( argvRead( sys.argv ) )







# =============================================================================
# Block 2 of functions
# =============================================================================

def polGenEMD( polNumb, beads_per_chain, bond_length, start_positions ):
    
    polymers = polymer.positions(n_polymers= polNumb,
                                 beads_per_chain=beads_per_chain,
                                 bond_length=bond_length, seed=round(random.random() * 1e+8) ,
                                 min_distance = bond_length, 
                                 start_positions = start_positions ,
                                 respect_constraints = True
                                 )    
    
    


def polGen( beads_per_chain, bond_length, cosDic ):

    startPos = cosDic['startPos']
    b = cosDic['boxl']
    polNumb = 1

    poll = np.ones(( polNumb, beads_per_chain, 3 ) ) * startPos # np.array([b/2.,b/2.,b/2.])
    ii = 1
    dpos = np.array([0, 0, bond_length])
    while ii < beads_per_chain:
        curPos = poll[0,ii-1,:]
        for iii in range( 6 * 5) :
            if ( np.linalg.norm( curPos + dpos - b/2. , axis=0 ) > b/2. -bond_length ) or ( np.any( np.linalg.norm( curPos + dpos - poll, axis=2) < bond_length ) ): 
                dang = np.random.random(2) * 2 * np.pi
                dpos = np.array([
                    np.sin( dang[0]) * np.cos( dang[1] ) ,
                    np.sin( dang[0]) * np.sin( dang[1] ) ,
                    np.cos( dang[0] )
                    ]) * bond_length
            else:
                poll[0,ii,:] = curPos + dpos
                ii = ii +1
                break
        
        if iii == 6 * 5 -1:
            # pdb.set_trace()
            ii = ii -1
            poll[0,ii,:] = np.ones((3)) * startPos
            # print( ii+1,' -> ', ii)
    
    # Check Sphere Constraint
    cond1 = np.all( np.linalg.norm( poll - b/2., axis=2) <= b/2. -bond_length )
    # Check polymer Constraint
    cond2 = np.allclose( np.linalg.norm( poll[0,1:,:] - poll[0,:-1,:], axis=1), bond_length)
    # Check polymer-polymer minima distance is respected
    dm = scy.spatial.distance_matrix( poll[0,:], poll[0,:])
    dm [ np.diag_indices( dm.shape[0]) ] = bond_length
    cond3 = not np.any( dm < bond_length )
    
    
    if cond1 and cond2 and cond3:
        print('Polymer generation successful!')
        return poll
    else :
        print('Error generating polymer: returning False')
        return False






def polGen2( bpcv, bond_length, cosDic ):

    b = cosDic['boxl']

    if type( bpcv) is not list:
        bpcv = [ bpcv]

    chkStaPos = (('startPosi' in cosDic.keys()) and (cosDic['startPosi'].shape[0] == len(bpcv)))
    if chkStaPos:
        pollnp = cosDic['startPosi']

    polly = []
    itertimes = 6 * 5 # coordination number * attempts
    for poli, bpc in enumerate(bpcv):

        ii = 1
        while ii < bpc:
            
            if ii == 1 and poli == 0:
                if not chkStaPos:
                    curPos = np.random.random((1,3)) * b
                    while np.linalg.norm( curPos  - b/2. , axis=1 ) > b/2. -bond_length : 
                        curPos = np.random.random((1,3)) * b
                    
                    poll = np.copy(curPos)
                    pollnp = np.copy(poll)    
                else:
                    poll = cosDic['startPosi'][ None, 0, :]
                    curPos = cosDic['startPosi'][ None, 0, :]
                
            elif ii == 1:
                if not chkStaPos:
                    curPos = np.random.random((1,3)) * b
                    while ( np.linalg.norm( curPos  - b/2. , axis=1 ) > b/2. -bond_length ) or ( np.any( np.linalg.norm( curPos - pollnp, axis=2) < bond_length ) ): 
                        curPos = np.random.random((1,3)) * b
                    
                    poll = np.copy(curPos)
                    pollnp = np.concatenate( (pollnp, poll), 0)
                else:
                    poll = cosDic['startPosi'][ None, poli, :]                    
                    curPos = cosDic['startPosi'][ None, poli, :]                    
                
            else:
                curPos = poll[ None,ii-1,:]
                
            # random direction
            dangThe = np.random.random() * 2 * np.pi
            dangPh = np.random.random() * np.pi
            dpos = np.array([
                np.sin( dangThe) * np.cos( dangPh ) ,
                np.sin( dangThe) * np.sin( dangPh ) ,
                np.cos( dangThe )
                ]) * bond_length
                
            
            for iii in range( itertimes) :
                if ( np.linalg.norm( curPos + dpos - b/2. , axis=1 ) > b/2. -bond_length ) or ( np.any( np.linalg.norm( curPos + dpos - pollnp, axis=1) < bond_length ) ): 
                    dangThe = np.random.random() * 2 * np.pi
                    dangPh = np.random.random() * np.pi
                    dpos = np.array([
                        np.sin( dangThe) * np.cos( dangPh ) ,
                        np.sin( dangThe) * np.sin( dangPh ) ,
                        np.cos( dangThe )
                        ]) * bond_length
                else:
                    pollnp = np.concatenate( ( pollnp, curPos + dpos), 0)
                    poll = np.concatenate( ( poll, curPos + dpos), 0)
                    ii = ii +1
                    break
            
            if iii == itertimes -1:
                # pdb.set_trace()
                ii = ii -1
                pollnp = np.delete( pollnp, -1, 0)
                poll = np.delete( poll, -1, 0)
                # print( ii+1,' -> ', ii)
        
        # Check Sphere Constraint
        cond1 = np.all( np.linalg.norm( poll - b/2., axis=1) <= b/2. -bond_length )
        # Check polymer Constraint
        cond2 = np.allclose( np.linalg.norm( poll[ 1:,:] - poll[ :-1,:], axis=1), bond_length)
        # Check polymer-polymer minimal distance is respected
        dm = scy.spatial.distance_matrix( poll, poll)
        dm [ np.diag_indices( dm.shape[0]) ] = bond_length
        cond3 = not np.any( dm < bond_length )
    
        polly += [poll]
    
    
        if cond1 and cond2 and cond3:
            print('Polymer',poli+1,'/',len(bpcv),'generation successful!')
        else :
            print('Error generating polymer: returning False')
            return False

    return polly






def genBS( Npart, mol_type, nbs = 3, seedbs = random.randint(0,10**9) ):
    
    Npol = Npart[-1]
    
    # nbs = 3
    p0, sp0 = 300, 120 # gaussian
    # x uniform
    s0, ss0 = 200, 100
    
    thp = np.zeros( ( len(mol_type), nbs))
    thx = np.zeros( ( len(mol_type), nbs))
    ths = np.zeros( ( len(mol_type), nbs))
    
    # this helps create a predictable common random distributions for parallel runs
    # WARNING: might need to be deprecated from numpy versions > 1.16 [check best practice on line]
    np.random.seed( seedbs )
    
    for mi in range( len(mol_type)) :
        thp[ mi, :] = np.minimum( np.maximum( np.random.normal( p0, sp0, nbs), 0), 600)
        thx[ mi, :] = np.random.randint( 2 * Npol ** (1/2), Npol - 2 * Npol ** (1/2), nbs) 
        ths[ mi, :] = np.maximum( np.random.normal( s0, ss0, nbs), 10)
    
    
    Rx = np.zeros( ( len(mol_type), Npol))
    
    for mi in range( len(mol_type)) :
        for pi in range( nbs) :
            Rx[ mi, :] = Rx[ mi, :] + scy.stats.norm.pdf( range(Npol), thx[ mi, pi], ths[ mi, pi]) * thp[ mi, pi]
            
    # zero out the Rx elements that are not the max for a given x
    Rxmask = Rx.max(axis=0, keepdims=1) == Rx
    Rx[ ~ Rxmask] = 0
    
    
    BS = np.clip( np.round( Rx * np.random.randint( 0, 2, Npol) ), 0, 1)
    
    np.mean(BS == 1)
    np.max( Rx * np.random.randint( 0, 2, Npol))
    np.std( Rx * np.random.randint( 0, 2, Npol))    
    
    
    bsl = []
    for mi in range( len(mol_type)) :
        BS[ mi, : np.int_( Npol ** (1/2) ) ] = 0
        BS[ mi, np.int_( -Npol ** (1/2) ) : ] = 0
        bsl = bsl + [  list(np.where( BS[ mi, :] == 1 )[0]) ]
    
    return bsl






def encodeHMM():
    chrHMM = pd.read_csv( strGenetics + '/wgEncodeBroadHmmHuvecHMM.bed.gz', sep='\t', header=None, compression='gzip')
    header = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']
    chrHMM.columns = header[:len(chrHMM.columns)]
       
    chromHMM_map = chrHMM.name.str.split('_', n=1, expand=True).drop_duplicates()
    chromHMM_map.columns = ['class_n','class_id']
    chromHMM_map['name'] = chrHMM.name.drop_duplicates()
    chromHMM_map.to_csv( strGeneticsHome + '/broadChromHMM_table.csv', sep='\t', index=False )
    # save_data or write_excel to be tested
    # chromHMM_map = pd.write_excel( strHome + '/epsBrackley2016.ods', engine="odf", sheet_name='encodeHMM', index_col=False)





def calcUnits( molC, N, chrend, chrstart, bond_length, eta=.05)   :
    # pdb.set_trace()
    # =============================================================================
    # Unit of measure
    # =============================================================================
    try:
        ll = chrend - chrstart
        res = ll  / N
    except:
        ll = [chrend[i] - chrstart[i] for i in range(len(chrend))]
        res = ll[0] / N[0]
        N = np.array( N ).sum()
        ll = np.array( ll ).sum()
    
    # 
    bioD = {'genome length': 
                {'hg19': 3.1 * 2 * 10**9  , 
                'mm9': 6.5 * 10**9 } , 
            'nucleus radius': # micron
                {'HeLa 1': (3 * 690 / (4 * np.pi) )**(1/3.)  , 
                'HeLa 2':  (3 * 374 / (4 * np.pi) )**(1/3.)  ,
                'Fibroblast':  (3 * 500 / (4 * np.pi) )**(1/3.)  ,
                'mESC Nicodemi': 3.5 / 2 ,  # this to be checked
                'mouse L cell': (3 * 435 / (4 * np.pi) )**(1/3.) } 
            }
    
    G = bioD['genome length']['hg19'] 
    R = bioD['nucleus radius']['Fibroblast'] * 10**(-6)
    
    # box size
    b = 2 * R * ( N / G * res) **(1/3.)  # meters
    
    # linearly depends on the estimation of nucleus radius!!!
    sig = ( res / ll ) ** ( 1/ 3.) * b  # meters
    # sig = ( res / G ) ** ( 1/ 3.) * R  # meters
    dunit = sig / bond_length
    
    r = b / 2
    
    # simul box size
    bsimsig = np.int_( b * 2 / dunit )
    rsimsig = np.int_( r * 2 / dunit )
    bsim = b * 2
    rsim = r * 2
    
    # time units
    nu = 10**(-2) # Pascal * sec [.1 P]
    KbT = 4.11 * 10 ** (-21) # T= 298K [Joule]
    gamm = 3 * np.pi * nu * sig
    D = KbT / gamm
    
    # LJ time 
    mkdalt = 1000 * 1.66033 * 10**-27 
    tLJ = sig * np.sqrt(  50 * mkdalt / KbT)
    
    
    # standard MD
    eta = .025 # .1 ?
    tau = eta * ( 6 * np.pi * sig**3 / KbT)
    
    # Browninan time scale
    tB = 3 * np.pi * sig ** 3 * nu / KbT # sig ** 2 / D [sec]
    # dramatically depends on the correct estimation of sigma!! 
    # 3 * np.pi * (3*10**-8) ** 3 * nu / KbT # sig ** 2 / D [sec]
    
    nts = 10**7 # number of simulation time steps
    ts = 10**-2 # simulation time step
    tT = nts * tB * ts # mapped time duration of simulation [sec]
    
    
    # pdb.set_trace()
    # map binder concentration
    Na = 6.022 * 10**23
    molPart = []
    for CnA in molC:
        # CnA = 10 # n mol / liter = 10**-9 mol / 10**-3 cubic meters
        P = np.int_( CnA * 10**-9 * Na * ( 4/ 3. * np.pi * rsim**3 * 10**3 ) )
        Ptot = np.int_( CnA * 10**-9 * Na * ( 4/ 3. * np.pi * R**3 * 10**3 ) )
    
        # given number of particles map concentration
        c = ( 1100 * 3 / ( 4 * np.pi) ) * ( rsim )**(-3) / Na # mol / m
        c2 = c * 10 ** (9) / 10** (3) # nano mol / liter    
    
        molPart += [P]
    
    # add num of pol beads as last element
    # molPart += [N]
    
    return molPart, bsimsig, tB, sig, tau, tLJ







def genBrackley2( bpres, chrname, chrstart, chrend, bpmin = 90 ):

    chrHMM = pd.read_csv( strGenetics + '/wgEncodeBroadHmmHuvecHMM.bed.gz', sep='\t', header=None, compression='gzip')
    header = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']
    chrHMM.columns = header[:len(chrHMM.columns)]
    chromHMM_map = pd.read_excel( strHome + '/modelBloom2021.ods', engine="odf", sheet_name='encodeHMM', index_col=False)

    chrHMM2 = chrHMM.merge( chromHMM_map, how='left', left_on='name', right_on='name' )
    
    # check mergin went good
    chrHMM2.class_id.unique()
    
    # check population distribution of classes
    chrHMM2.groupby('class_id').count()
    chrHMM2.groupby('brackley2016').count()
    
    
    chrHMM2['dchrbp'] = chrHMM2['chromEnd'] - chrHMM2['chromStart']
    chrHMM2 = chrHMM2[ chrHMM2['dchrbp'] >= bpmin ]
    
    
    
    chrsub = chrHMM2[ (chrHMM2['chrom']== chrname) & ( chrHMM2['chromStart'] > chrstart) & ( chrHMM2['chromEnd'] < chrend) ]
    
    
    part = np.arange( chrsub['chromStart'].values[0], chrsub['chromEnd'].values[-1], bpres )
    partdf = pd.DataFrame( part, columns=['part'])
    
    
    part2 = pd.merge_asof( chrsub[['chromStart','name','brackley2016']], partdf, left_on = 'chromStart', right_on ='part' ,  tolerance= bpres )
    part2['partn'] = (part2['part'] / bpres)# .astype( np.int32)
    
    
    #
    part2['dclass']= part2.shift(-1)['brackley2016'] - part2['brackley2016']
    part4 = part2[ part2['dclass'] != 0]
    
    part4['dpn'] = part4.shift(-1)['partn'] - part4['partn']
    part4['dpn2'] = part4['dpn']
    
    part4['dpn2'].iloc[-1] = np.round( (chrsub['chromEnd'].values[-1] - part4['chromStart'].values[-1]) / bpres )
    part4.replace({'dpn2':{0:1}}, inplace=True)
    part4['dpn2'] = part4['dpn2'].astype('int')
    
    
    # checks
# =============================================================================
#     part4['dpn'].sum() * bpres
#     part4['dpn2'].sum() * bpres
#     part4['dpn2'].sum()
# =============================================================================
    
    
    part4['partn3'] = part4['dpn2'].cumsum().shift(+1, fill_value=0) # part4.partn2.values - part4 ['partn2'].iloc[0]
    
    bs = np.zeros( ( len( part4.brackley2016.unique()) , part4['dpn2'].sum() ) ) * np.nan
    for idx, parti in part4.reset_index().iterrows() :
        bstmp = np.arange( parti.partn3, parti.partn3 + parti.dpn2 )
        bs[ parti.brackley2016, bstmp ] = bstmp
    
    
    part4['dstart']= part4.shift(-1)['chromStart'] - part4['chromStart']
    part4['dstartn']= part4['dstart'] / bpres
    
    part4[['chromStart','brackley2016','partn3','dpn2','dstartn']]
    
    
    binding = []
    for bsi in range( bs.shape[0]-1 ):
        bstmp = bs[ bsi, :]
        binding += [list( bstmp[ ~np.isnan( bstmp)] + np.int_( bs.shape[1] ** (1/2) ) )]    

    npol = 2 * np.int_( bs.shape[1] ** (1/2) ) + bs.shape[1]


    return binding, npol








def genBrackley2_bin( bpres, chrname, chrstart, chrend, tailtyp, strpart = 'hicpart', bpmin = 90 ):

    # tailtyp = 5 # model.oldpol_type
    # strpart = 'hicpart' # given # natural
    
    chrHMM = pd.read_csv( strGenetics + '/wgEncodeBroadHmmHuvecHMM.bed.gz', sep='\t', header=None, compression='gzip')
    header = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']
    chrHMM.columns = header[:len(chrHMM.columns)]
    chromHMM_map = pd.read_excel( strHome + '/modelBloom2021.ods', engine="odf", sheet_name='HMM2model', index_col=False)
    
    chrHMM2 = chrHMM.merge( chromHMM_map, how='left', left_on='name', right_on='name' )
    
   
    chrHMM2['dchrbp'] = chrHMM2['chromEnd'] - chrHMM2['chromStart']
    chrHMM2 = chrHMM2[ chrHMM2['dchrbp'] >= bpmin ]
    
    
    ##
    try:
        chrL = list(zip( chrname, chrstart, chrend))
    except:
        chrL = [ (chrname, chrstart, chrend) ]
        
    ##
    polid, npol, taillen = 0, [], []
    binning, bsmap = pd.DataFrame([]), pd.DataFrame([])
    for chrname, chrstart, chrend in chrL:
        print( 'Polymer: ', chrname, chrstart, chrend)

        # this selects the region 
        chrsub = chrHMM2[ (chrHMM2['chrom']== chrname) & ( chrHMM2['chromStart'] >= chrstart) & ( chrHMM2['chromEnd'] <= chrend) ]
        
        # create partition
        if strpart == 'natural':
            part = np.arange( chrsub['chromStart'].values[0], chrsub['chromEnd'].values[-1], bpres )
        elif strpart == 'hicpart':
            # map into the hic matrix
            part = np.int_( np.arange( 
                np.round( chrsub['chromStart'].values[0] / bpres ) * bpres , 
                np.round( chrsub['chromEnd'].values[-1] / bpres ) * bpres +1, 
                bpres
                ) )
        elif strpart == 'given':
            part = partgiven
            
        
        partdf = pd.DataFrame( part, columns=['part'])
        
        
        # merge as of
        part2 = pd.merge_asof( chrsub[['chromStart','name','brackley2016']], partdf, 
                              left_on = 'chromStart', right_on ='part' ,  
                              tolerance= bpres, direction='nearest' )
        
        part2['partn'] = (part2['part'] / bpres)# .astype( np.int32)
        
        
        
        # remove classes that are contiguous and identical (this can happen because we remove intervals below threshold bpmin )
        part2['dclass']= part2.shift(-1)['brackley2016'] - part2['brackley2016']
        part4 = part2[ part2['dclass'] != 0]
        
        # calculate how many beads are in each interval
        part4['dpn'] = part4.shift(-1)['partn'] - part4['partn']
        part4['dpn2'] = part4['dpn']
        
        # deal with last interval
        part4['dpn2'].iloc[-1] = np.round( (chrsub['chromEnd'].values[-1] - part4['chromStart'].values[-1]) / bpres )
        part4['dpn'].iloc[-1] = part4['dpn2'].iloc[-1]
        
        # assign 1 bead to all intervals smaller than bpres
        part4.replace({'dpn2':{0:1}}, inplace=True)
        part4['dpn2'] = part4['dpn2'].astype('int')
        
        
        # calculate how many beads are there
        part4['cumpos'] = part4['dpn2'].cumsum() # .shift(+1, fill_value=0) # part4.partn2.values - part4 ['partn2'].iloc[0]
        bslen = part4['dpn2'].sum()
        # estimate tails of polymer
        taillentmp = np.int_( np.round( bslen ** (1/2) ) )
        npoltmp = 2 * taillentmp + bslen
        
        
        # create binning from partition
        taildf = pd.DataFrame(
            np.zeros( ( 1,  part4.shape[1])) * np.nan ,
            columns = part4.columns
            )
        
        taildf['brackley2016'] = int( tailtyp )
        taildf['dpn2'] = taillentmp
        taildf['dpn'] = taillentmp
        
        part4 = taildf.append( part4.append( taildf))
        
        
        
        
        
        
        # create binning from partition
        binntmp = partdf.merge( part4, how= 'left', on='part')
        binntmp['brackley2016'] = binntmp['brackley2016'].fillna( method='ffill')
        binntmp['type'] = binntmp['brackley2016']
        
        if binntmp.dpn2.sum() != binntmp.shape[0]:
            print('This is not good! binning.dpn2.sum() != binning.shape[0]')
            raise 
            
        taildf = pd.DataFrame(
            np.zeros( ( 1,  binntmp.shape[1])) * np.nan ,
            columns = binntmp.columns
            )
        
        taildf['type'] = int( tailtyp )
        taildf['part'] = - 2 * (polid+1) 
        binntmp = binntmp.append( taildf)

        taildf['part'] = - 2 * (polid+1) +1
        binntmp = taildf.append( binntmp )
            
            
        
        
        # map of bs
        bsmaptmp = part4[['brackley2016','dpn2','chromStart']]
        bsmaptmp.columns = ['type','bsbatch','chromStart']
        
        
        
        ###
        binntmp['polyid'] = polid
        binning = binning.append( binntmp)
        
        ###
        bsmaptmp['polyid'] = polid
        bsmap = bsmap.append( bsmaptmp)

        ###        
        polid = polid + 1
        taillen += [taillentmp]
        npol += [npoltmp]
    
    
    
    return npol, taillen, binning, bsmap







def genHMM_bin( model, strpart = 'hicpart', bpmin = 90 ):

    # tailtyp = 5 # model.oldpol_type
    # strpart = 'hicpart' # given # natural
    
    chrHMM = pd.read_csv( strGenetics + '/wgEncodeBroadHmmHuvecHMM.bed.gz', sep='\t', header=None, compression='gzip')
    header = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']
    chrHMM.columns = header[:len(chrHMM.columns)]
    chromHMM_map = pd.read_excel( strHome + '/' + model.modelods + '_model.ods', engine="odf", sheet_name= model.modelods_modelsheet, index_col=False)
    
    chrHMM2 = chrHMM.merge( chromHMM_map, how='left', left_on='name', right_on='name' )
    
   
    chrHMM2['dchrbp'] = chrHMM2['chromEnd'] - chrHMM2['chromStart']
    chrHMM2 = chrHMM2[ chrHMM2['dchrbp'] >= bpmin ]
    
    
    ##
    try:
        chrL = list(zip( model.chrname, model.chrstart, model.chrend))
    except:
        chrL = [ (model.chrname, model.chrstart, model.chrend) ]
        
    ##
    polid, npol, taillen = 0, [], []
    binning, bsmap = pd.DataFrame([]), pd.DataFrame([])
    for chrname, chrstart, chrend in chrL:
        print( 'Polymer: ', chrname, chrstart, chrend)

        # this selects the region 
        chrsub = chrHMM2[ (chrHMM2['chrom']== chrname) & ( chrHMM2['chromStart'] >= chrstart) & ( chrHMM2['chromEnd'] <= chrend) ]
        

        
        
        # create partition
        if strpart == 'natural':
            part = np.arange( chrsub['chromStart'].values[0], chrsub['chromEnd'].values[-1], model.bpres )
        elif strpart == 'hicpart':
            # map into the hic matrix
            part = np.int_( np.arange( 
                np.round( chrsub['chromStart'].values[0] / model.bpres ) * model.bpres , 
                np.round( chrsub['chromEnd'].values[-1] / model.bpres ) * model.bpres +1, 
                model.bpres
                ) )
        elif strpart == 'given':
            part = partgiven
            
        
        partdf = pd.DataFrame( part, columns=['part'])
        
        
        # merge as of
        # pdb.set_trace()
        part2 = pd.merge_asof( chrsub[['chromStart','name', model.modelname ]], partdf, 
                              left_on = 'chromStart', right_on ='part' ,  
                              tolerance= model.bpres, direction='nearest' )
        
        part2['partn'] = (part2['part'] / model.bpres)# .astype( np.int32)
        
        
        
        # remove classes that are contiguous and identical (this can happen because we remove intervals below threshold bpmin )
        part2['dclass']= part2.shift(-1)[ model.modelname ] - part2[ model.modelname ]
        part4 = part2[ part2['dclass'] != 0]
        
        # calculate how many beads are in each interval
        part4['dpn'] = part4.shift(-1)['partn'] - part4['partn']
        part4['dpn2'] = part4['dpn']
        
        # deal with last interval
        part4['dpn2'].iloc[-1] = np.round( (chrsub['chromEnd'].values[-1] - part4['chromStart'].values[-1]) / model.bpres )
        part4['dpn'].iloc[-1] = part4['dpn2'].iloc[-1]
        
        # assign 1 bead to all intervals smaller than bpres
        part4.replace({'dpn2':{0:1}}, inplace=True)
        part4['dpn2'] = part4['dpn2'].astype('int')
        
        
        # calculate how many beads are there
        part4['cumpos'] = part4['dpn2'].cumsum() # .shift(+1, fill_value=0) # part4.partn2.values - part4 ['partn2'].iloc[0]
        bslen = part4['dpn2'].sum()
        # estimate tails of polymer
        taillentmp = np.int_( np.round( bslen ** (1/2) ) )
        npoltmp = 2 * taillentmp + bslen
        
        
        # create binning from partition
        taildf = pd.DataFrame(
            np.zeros( ( 1,  part4.shape[1])) * np.nan ,
            columns = part4.columns
            )
        
        taildf[ model.modelname ] = int( model.tailtyp )
        taildf['dpn2'] = taillentmp
        taildf['dpn'] = taillentmp
        
        part4 = taildf.append( part4.append( taildf))
        
        
        
        
        
        
        # create binning from partition
        binntmp = partdf.merge( part4, how= 'left', on='part')
        binntmp[ model.modelname ] = binntmp[ model.modelname ].fillna( method='ffill')
        binntmp['type'] = binntmp[ model.modelname ]
        
        if binntmp.dpn2.sum() != binntmp.shape[0]:
            print('This is not good! binning.dpn2.sum() != binning.shape[0]')
            raise 
            
        taildf = pd.DataFrame(
            np.zeros( ( 1,  binntmp.shape[1])) * np.nan ,
            columns = binntmp.columns
            )
        
        taildf['type'] = int( model.tailtyp )
        taildf['part'] = - 2 * (polid+1) 
        binntmp = binntmp.append( taildf)

        taildf['part'] = - 2 * (polid+1) +1
        binntmp = taildf.append( binntmp )
            
            
        
        
        # map of bs
        bsmaptmp = part4[[ model.modelname ,'dpn2','chromStart']]
        bsmaptmp.columns = ['type','bsbatch','chromStart']
        
        
        
        ###
        binntmp['polyid'] = polid
        binning = binning.append( binntmp)
        
        ###
        bsmaptmp['polyid'] = polid
        bsmaptmp['monocumsum'] = bsmaptmp['bsbatch'].cumsum()
        bsmap = bsmap.append( bsmaptmp)

        ###        
        polid = polid + 1
        taillen += [taillentmp]
        npol += [npoltmp]
    
    
    
    return npol, taillen, binning, bsmap









def readSV( model ):

    # model.resBestFN = '/SA/resltBest_'+.regg+'_N'+str(N)+'_M'+str(M)+'.csv'
    
    ##
    try:
        chrL = list(zip( model.chrname, model.chrstart, model.chrend))
    except:
        chrL = [ (model.chrname, model.chrstart, model.chrend) ]
        
    ##
    polid, npol, taillen = 0, [], []
    binning, bsmap, bestpd = pd.DataFrame([]), pd.DataFrame([]), pd.Series([])
    for chrid, (chrname, chrstart, chrend) in enumerate(chrL):
        
        # save/read best corr matrix
        # ttcpd = pd.read_csv( model.resBestFN[chrid]
        #                        , sep = ',', index_col=False )
        
        # model.N = model.ttcpd.N[0]
        # model.M = model.ttcpd.M[0]
        # model.R = np.copy(model.M)
        # model.NR = model.N * model.R
        # pdb.set_trace()
        # select one ttc
        # model.besttmp = model.ttcpd.iloc[ model.bestIndex[chrid] ]
        besttmp = model.besttmp[chrid]
        besttmp['polyid'] = polid
        bestpd = bestpd.append( besttmp)
        
        # ttcpd2 = ttcpd[ [ str(ni) for ni in range(0, model.NR) ] ]
        # (ttcpd2 != 0).sum(1).min(), (ttcpd2 != 0).sum(1).max()
        
        ttcsel = besttmp[ [ str(ni) for ni in range(0, model.NR) ] ].values.flatten()
        


        ###
        Pimap = pd.DataFrame( np.repeat( np.linspace( chrstart, chrend-model.bpres, int((chrend - chrstart) / model.bpres) ), model.R) )
        Pimap['chr'] = chrname
        Pimap.columns = ['chrstart','chr']
        Pimap['bs'] = ttcsel
        Pimap['Idx'] = np.array( list(range(0,model.R)) * model.N )
        #
        Pimapsub = Pimap.merge( model.visMapOrient, how='right', on=['chr','chrstart','Idx'])
        #
        ttcsel = Pimapsub.bs.values
        NR = ttcsel.size
        # chrend = int(Pimapsub.shape[0] / model.R * model.bpres)
        # chrstart = 0
        
        
        ###
        binntmp = pd.DataFrame( data={
            'cumpos': list(range(1,NR + 1)) ,
            'type' : ttcsel + model.M-1 ,
            'part' : Pimapsub.chrstart ,
            'chromStart' : Pimapsub.chrstart
            })
        
        taildf = pd.DataFrame(
            np.zeros( ( 1,  binntmp.shape[1])) * np.nan ,
            columns = binntmp.columns
            )
        
        taildf['type'] = 2 * model.M -1
        taildf['part'] = - 2 * (polid+1) 
        binntmp = binntmp.append( taildf)

        taildf['part'] = - 2 * (polid+1) +1
        binntmp = taildf.append( binntmp )
        
        
        # map of bs        
        binntmp['Index'] = binntmp.reset_index().index
        binntmp['dclass']= binntmp.shift(-1)[ 'type' ] - binntmp[ 'type' ]
        bsmaptmp = binntmp[ binntmp['dclass'] != 0]        
        bsmaptmp['bsbatch'] = bsmaptmp['Index'] - bsmaptmp.shift(+1)['Index']
        
        
        # estimate tails of polymer
        taillentmp = np.int_( np.round( NR ** (1/2) ) )
        
        npoltmp = NR
        
        #
        bsmaptmp['bsbatch'].iloc[0] = taillentmp
        bsmaptmp['bsbatch'].iloc[-1] = taillentmp


        ###
        binntmp['polyid'] = polid
        binning = binning.append( binntmp)
        
        ###
        bsmaptmp['polyid'] = polid
        bsmaptmp['bsbatch'] = bsmaptmp['bsbatch'].astype(int)
        bsmaptmp['monocumsum'] = bsmaptmp['bsbatch'].cumsum()
        bsmap = bsmap.append( bsmaptmp)


        
        ###        
        polid = polid + 1
        taillen += [taillentmp]
        npol += [npoltmp]



    
    return npol, taillen, binning, bsmap, bestpd








def readClasses( model ):

    # model.resBestFN = '/SA/resltBest_'+.regg+'_N'+str(N)+'_M'+str(M)+'.csv'
    
    ##
    try:
        chrL = list(zip( model.chrname, model.chrstart, model.chrend))
    except:
        chrL = [ (model.chrname, model.chrstart, model.chrend) ]
        
    ##
    polid, npol, taillen = 0, [], []
    binning, bsmap, bestpd = pd.DataFrame([]), pd.DataFrame([]), pd.Series([])
    for chrid, (chrname, chrstart, chrend) in enumerate(chrL):
        
        # save/read best corr matrix
        # ttcpd = pd.read_csv( model.resBestFN[chrid]
        #                        , sep = ',', index_col=False )
        
        # model.N = model.ttcpd.N[0]
        # model.M = model.ttcpd.M[0]
        # model.R = np.copy(model.M)
        # model.NR = model.N * model.R
        # pdb.set_trace()
        # select one ttc
        # model.besttmp = model.ttcpd.iloc[ model.bestIndex[chrid] ]
        besttmp = model.besttmp[chrid]
        besttmp['polyid'] = polid
        bestpd = bestpd.append( besttmp)
        
        # ttcpd2 = ttcpd[ [ str(ni) for ni in range(0, model.NR) ] ]
        # (ttcpd2 != 0).sum(1).min(), (ttcpd2 != 0).sum(1).max()
        
        ttcsel = besttmp[ [ str(ni) for ni in range(0, model.NR) ] ].values.flatten()
        
        # Entropy 2
        # uni2 = np.unique( ttcsel.tolist()+list(range(0, model.M)), return_counts=True)
        # ncou2 = uni2[1]-1
        # nint = ncou2[1:].sum()

        binntmp = pd.DataFrame( data={
            'cumpos': list(range(1,model.NR + 1)) ,
            'type' : ttcsel + model.M-1 ,
            'part' : np.arange(chrstart, chrend, (chrend-chrstart)/model.NR)[: model.NR] ,
            'chromStart' : np.arange(chrstart, chrend, (chrend-chrstart)/model.NR)[: model.NR]
            })
        
        taildf = pd.DataFrame(
            np.zeros( ( 1,  binntmp.shape[1])) * np.nan ,
            columns = binntmp.columns
            )
        
        taildf['type'] = 2 * model.M -1
        taildf['part'] = - 2 * (polid+1) 
        binntmp = binntmp.append( taildf)

        taildf['part'] = - 2 * (polid+1) +1
        binntmp = taildf.append( binntmp )
        
        
        # map of bs        
        binntmp['Index'] = binntmp.reset_index().index
        binntmp['dclass']= binntmp.shift(-1)[ 'type' ] - binntmp[ 'type' ]
        bsmaptmp = binntmp[ binntmp['dclass'] != 0]        
        bsmaptmp['bsbatch'] = bsmaptmp['Index'] - bsmaptmp.shift(+1)['Index']
        
        
        # estimate tails of polymer
        taillentmp = np.int_( np.round( model.NR ** (1/2) ) )
        
        npoltmp = model.NR
        
        #
        bsmaptmp['bsbatch'].iloc[0] = taillentmp
        bsmaptmp['bsbatch'].iloc[-1] = taillentmp


        ###
        binntmp['polyid'] = polid
        binning = binning.append( binntmp)
        
        ###
        bsmaptmp['polyid'] = polid
        bsmaptmp['bsbatch'] = bsmaptmp['bsbatch'].astype(int)
        bsmaptmp['monocumsum'] = bsmaptmp['bsbatch'].cumsum()
        bsmap = bsmap.append( bsmaptmp)


        
        ###        
        polid = polid + 1
        taillen += [taillentmp]
        npol += [npoltmp]



    
    return npol, taillen, binning, bsmap, bestpd












def readClassesScal( model, minnbin=5 ):

    # model.resBestFN = '/SA/resltBest_'+.regg+'_N'+str(N)+'_M'+str(M)+'.csv'
    
    ##
    try:
        chrL = list(zip( model.chrname, model.chrstart, model.chrend))
    except:
        chrL = [ (model.chrname, model.chrstart, model.chrend) ]
        
    ##
    polid, npol, taillen = 0, [], []
    binning, bsmap, bestpd = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
    for chrid, (chrname, chrstart, chrend) in enumerate(chrL):
        
        # save/read best corr matrix
        # ttcpd = pd.read_csv( model.resBestFN[chrid]
        #                         , sep = ',', index_col=False )
        
        # model.N = model.ttcpd.N[0]
        # model.M = model.ttcpd.M[0]
        # model.R = np.copy(model.M)
        # model.NR = model.N * model.R

        # select one ttc
        # besttmp = model.ttcpd.iloc[ model.bestIndex[chrid] ]
        besttmp = model.besttmp[chrid]
        besttmp['polyid'] = polid
        bestpd = bestpd.append( besttmp)
        
        # ttcpd2 = ttcpd[ [ str(ni) for ni in range(0, model.NR) ] ]
        # (ttcpd2 != 0).sum(1).min(), (ttcpd2 != 0).sum(1).max()
        
        ttcsel = besttmp[ [ str(ni) for ni in range(0, model.NR) ] ].values.flatten()
        
        
        # model.scalstep = 4
        multipart = np.repeat( np.arange( 0, np.ceil( model.NR / model.scalstep ), 1), model.scalstep )[ : model.NR]
        # chrompart = multipart * int((model.chrend-model.chrstart)/model.NR) * model.scalstep + model.chrstart
        
        #
        binntmpo = pd.DataFrame( data={
            # 'cumpos': list(range(1,model.NR + 1)) ,
            'type' : ttcsel + model.M-1 ,
            'part' : multipart ,
            # 'chromStart' : chrompart
            })
        binntmpo['polyid'] = polid
        
        # 
        # pdb.set_trace()
        cass, partBeads = bin2type2( binntmpo )
        # pdb.set_trace()
        partBeads =partBeads.reset_index()
        partBeads['complextypes'] = partBeads.reset_index()[0].astype(str)
 
       


        if True:
            partBeads['oldcomplextypes'] = partBeads[0]
            uniqlist = np.unique( partBeads.complextypes, return_counts=True)
            uniqlist2 = np.unique( partBeads[0], return_counts=True)[0].tolist()
        
            acceptlist = np.array(uniqlist2)[ uniqlist[1] > minnbin].tolist()
            negllist = np.array(uniqlist2)[ uniqlist[1] <= minnbin].tolist()
            
            
            
            basetyp = np.concatenate( [ 
                np.ones((model.M-1,model.scalstep-1),dtype=np.int_) * (model.M-1), 
                np.arange( model.M, 2*model.M-1,dtype=np.int_)[:,None]
                ], axis=1).tolist()


            acceptarr = pd.DataFrame(acceptlist).append( pd.DataFrame(basetyp)).drop_duplicates().values # .tolist()
            acceptarr = acceptarr[~np.alltrue([model.M-1] * model.scalstep == acceptarr, axis=1)]


            nppartb = np.array(partBeads[0].values.tolist())
            # 
            for vec2 in negllist:
                # cas2 = negllist[0]
                uni2 = np.unique(np.int_( vec2), return_counts=True)
                ncou2 = np.zeros((model.M), dtype=np.int_)
                ncou2[ uni2[ 0]-model.M+1] = uni2[ 1]
                # remove grey color from calculation
                ncou2 = ncou2[1:]
        
                dd = np.zeros(acceptarr.shape[0])
                for idvec, vec in enumerate( acceptarr):
                    
                    uni = np.unique(np.int_(vec), return_counts=True)
                    ncou = np.zeros((model.M), dtype=np.int_)
                    ncou[ uni[ 0]-model.M+1] = uni[ 1]            
                    # remove grey color from calculation
                    ncou = ncou[1:]
        
                    v11 = np.int_(ncou > 0)
                    v21 = np.int_(ncou2 > 0)
                    
                    dd[idvec]= ( ((ncou2 - ncou)**2).sum() + ((v21 - v11)**2).sum() )**(1/2)
                    # (((v21 - v11)**2).sum() )**(1/2)
                    # ( ((ncou2 - ncou)**2).sum() )**(1/2)
                
                # pdb.set_trace()
                ddmin = np.min(dd)
                
                idddminv = np.argwhere(dd == ddmin).flatten()
                idddmin = np.random.choice( idddminv, size=1)
    
                minvec = acceptarr[ idddmin, : ]
                
                ##
                
                partmask = partBeads.complextypes == str( vec2)
                nppartb[ partmask, :] = minvec[0]
                
                
                # partBeads.complextypes [partBeads.complextypes == str( vec2)] = str(minvec[0])
                # minvecmask = np.alltrue( np.array(vec2) == np.array(partBeads[0].values.tolist()  ), 1)
                # partBeads[ pd.DataFrame(minvecmask)[0] ][[0]] = list(minvec[0])

            partBeads = partBeads[['polyid','part','oldcomplextypes']]
            partBeads['complextypeslist'] = nppartb.tolist()
            partBeads['complextypes'] = partBeads['complextypeslist'].astype(str)

            # partBeads[['complextypes']].apply( str2list, axis=1)
            # pdb.set_trace()


            binntmp = partBeads[['polyid','part','complextypes']]
            
            
            ###
            uniqlist = np.unique( partBeads.complextypes, return_counts=True)
            uniqlist2 = np.unique( partBeads['complextypeslist'], return_counts=True)[0].tolist()
                    
            
            # minnbin = 5
            typemap = pd.DataFrame( data={
                'complextypeslist': np.array(uniqlist2).tolist() , 
                'complextypes' : uniqlist[0] ,
                'type' : list(range( model.M-1, model.M-1 + len(uniqlist[1])))
                } )             
            
            
            
        else:
            uniqlist = np.unique( partBeads.complextypes, return_counts=True)
            uniqlist2 = np.unique( partBeads[0], return_counts=True)[0].tolist()

            # minnbin = 5
            typemap = pd.DataFrame( data={
                'complextypeslist': np.array(uniqlist2)[ uniqlist[1] > minnbin].tolist() , 
                'complextypes' : uniqlist[0][ uniqlist[1] > minnbin] ,
                'type' : list(range( model.M-1, model.M-1 + (uniqlist[1] > minnbin).sum()))
                } ) 
            
            binntmp['complextypes'] = partBeads.complextypes.where( np.isin( partBeads.complextypes, uniqlist[0][ uniqlist[1] > minnbin]), str( [model.M-1] * model.scalstep ) )

        


        binntmp = binntmp.merge( typemap[['type','complextypes']], how='left', on='complextypes')

        # binntmp = partBeads.merge( typemap[['type','complextypes']], how='left', on='complextypes')
        pollen = binntmp.shape[0]
        binntmp['chromStart'] = list(range(
            model.chrstart, model.chrend, int(np.ceil((model.chrend-model.chrstart)/model.NR) * model.scalstep)
            ))[ :pollen ]
        binntmp['cumpos'] = list(range(1, pollen+1))


        # tails        
        taildf = pd.DataFrame(
            np.zeros( ( 1,  binntmp.shape[1])) * np.nan ,
            columns = binntmp.columns
            )
        tailtype = typemap.type.unique()[-1] +1
        
        taildf['type'] = tailtype
        taildf['part'] = - 2 * (polid+1) 
        binntmp = binntmp.append( taildf)

        taildf['part'] = - 2 * (polid+1) +1
        binntmp = taildf.append( binntmp )
        
        
        # map of bs        
        binntmp['Index'] = binntmp.reset_index().index
        binntmp['dclass']= binntmp.shift(-1)[ 'type' ] - binntmp[ 'type' ]
        bsmaptmp = binntmp[ binntmp['dclass'] != 0]        
        bsmaptmp['bsbatch'] = bsmaptmp['Index'] - bsmaptmp.shift(+1)['Index']
        
        
        # estimate tails of polymer + laength of polymer
        taillentmp = np.int_( np.round( pollen ** (1/2) ) )

        npoltmp = int(model.NR / model.scalstep)

        #
        bsmaptmp['bsbatch'].iloc[0] = taillentmp
        bsmaptmp['bsbatch'].iloc[-1] = taillentmp


        ###
        binntmp['polyid'] = polid
        binning = binning.append( binntmp)
        
        ###
        bsmaptmp['polyid'] = polid
        bsmaptmp['bsbatch'] = bsmaptmp['bsbatch'].astype(int)
        bsmaptmp['monocumsum'] = bsmaptmp['bsbatch'].cumsum()
        bsmap = bsmap.append( bsmaptmp)


        
        ###        
        polid = polid + 1
        taillen += [taillentmp]
        npol += [npoltmp]



    
    return npol, taillen, binning, bsmap, bestpd, typemap






def str2list( stri):
    # print( stri[1:-1])
    # pdb.set_trace()
    coll = [ int(ele) for ele in stri[0].split() ]
    return coll









def genHMM_bin_frac( model, strpart = 'hicpart', bpmin = 90, fracd=None ):

    # tailtyp = 5 # model.oldpol_type
    # strpart = 'hicpart' # given # natural
    
    chrHMM = pd.read_csv( env.strGenetics + '/wgEncodeBroadHmmHuvecHMM.bed.gz', sep='\t', header=None, compression='gzip')
    header = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']
    chrHMM.columns = header[:len(chrHMM.columns)]
    chromHMM_map = pd.read_excel( env.strHome + '/' + model.modelods + '_model.ods', engine="odf", sheet_name= model.modelods_modelsheet, index_col=False)
    
    chrHMM2 = chrHMM.merge( chromHMM_map, how='left', left_on='name', right_on='name' )
    
   
    chrHMM2['dchrbp'] = chrHMM2['chromEnd'] - chrHMM2['chromStart']
    chrHMM2 = chrHMM2[ chrHMM2['dchrbp'] >= bpmin ]
    
    
    ##
    try:
        chrL = list(zip( model.chrname, model.chrstart, model.chrend))
    except:
        chrL = [ (model.chrname, model.chrstart, model.chrend) ]
        
    ##
    polid, npol, taillen = 0, [], []
    binning, bsmap = pd.DataFrame([]), pd.DataFrame([])
    for chrname, chrstart, chrend in chrL:
        print( 'Polymer: ', chrname, chrstart, chrend)

        # this selects the region 
        chrsub = chrHMM2[ (chrHMM2['chrom']== chrname) & ( chrHMM2['chromStart'] >= chrstart) & ( chrHMM2['chromEnd'] <= chrend) ]
        

        
        
        # create partition
        if strpart == 'natural':
            part = np.arange( chrsub['chromStart'].values[0], chrsub['chromEnd'].values[-1], model.bpres )
        elif strpart == 'hicpart':
            # map into the hic matrix
            part = np.int_( np.arange( 
                np.round( chrsub['chromStart'].values[0] / model.bpres ) * model.bpres , 
                np.round( chrsub['chromEnd'].values[-1] / model.bpres ) * model.bpres +1, 
                model.bpres
                ) )
        elif strpart == 'given':
            part = partgiven
            
        
        partdf = pd.DataFrame( part, columns=['part'])
        
        
        # merge as of
        # pdb.set_trace()
        part2 = pd.merge_asof( chrsub[['chromStart','name', model.modelname ]], partdf, 
                              left_on = 'chromStart', right_on ='part' ,  
                              tolerance= model.bpres, direction='nearest' )
        
        part2['partn'] = (part2['part'] / model.bpres)# .astype( np.int32)
        part2[ model.modelname ] = part2[ model.modelname ].astype('int')
        
        # remove classes that are contiguous and identical (this can happen because we remove intervals below threshold bpmin )
        part2['dclass']= part2.shift(-1)[ model.modelname ] - part2[ model.modelname ]
        part4 = part2[ part2['dclass'] != 0]        
        
        # pdb.set_trace()
        # if you want to remove a fraction
        if fracd != None:
            
            # create binning from partition (1)
            binntmp = partdf.merge( part4, how= 'left', on='part')
            binntmp[ model.modelname ] = binntmp[ model.modelname ].fillna( method='ffill')
            binntmp[ 'name' ] = binntmp[ 'name' ].fillna( method='ffill') ##
            binntmp['type'] = binntmp[ model.modelname ]        
            for fracdi in fracd:
                idrep = binntmp[ np.isin( binntmp.type, fracdi['type']) ].sample(frac=fracdi['frac'], random_state=fracdi['rseed']).index
                binntmp.type.loc[ idrep] = fracdi['typenew'] 
                binntmp.name.loc[ idrep] = fracdi['newname'] 
            #
            # pdb.set_trace()
            ## part2 = binntmp.dropna(axis=0, subset=['name'] )
            binntmp['dclass']= binntmp.shift(-1)[ 'type' ] - binntmp[ 'type' ] ##
            part4 = binntmp[ binntmp['dclass'] != 0] ##
            #             
            part4[ model.modelname ] = part4.type ##
            part4.drop('type', 1, inplace=True) ##
            #
            part4['partn'] = (part4['part'] / model.bpres)# .astype( np.int32)
            part4[ model.modelname ] = part4[ model.modelname ].astype('int')

            #             
            ## part2[ model.modelname ] = part2.type
            ## part2.drop('type', 1, inplace=True)
            #
            ## part2['partn'] = (part2['part'] / model.bpres)# .astype( np.int32)
            ## part2[ model.modelname ] = part2[ model.modelname ].astype('int')


        
        # remove classes that are contiguous and identical (this can happen because we remove intervals below threshold bpmin )
        ## part2['dclass']= part2.shift(-1)[ model.modelname ] - part2[ model.modelname ]
        ## part4 = part2[ part2['dclass'] != 0]
        
        # calculate how many beads are in each interval
        part4['dpn'] = part4.shift(-1)['partn'] - part4['partn']
        part4['dpn2'] = part4['dpn']
        
        # deal with last interval
        part4['dpn2'].iloc[-1] = np.round( (chrsub['chromEnd'].values[-1] - part4['chromStart'].values[-1]) / model.bpres )
        part4['dpn'].iloc[-1] = part4['dpn2'].iloc[-1]
        
        # assign 1 bead to all intervals smaller than bpres
        part4.replace({'dpn2':{0:1}}, inplace=True)
        part4['dpn2'] = part4['dpn2'].astype('int')
        
        
        # calculate how many beads are there
        part4['cumpos'] = part4['dpn2'].cumsum() # .shift(+1, fill_value=0) # part4.partn2.values - part4 ['partn2'].iloc[0]
        bslen = part4['dpn2'].sum()
        # estimate tails of polymer
        taillentmp = np.int_( np.round( bslen ** (1/2) ) )
        npoltmp = 2 * taillentmp + bslen
        
        
        # create binning from partition
        taildf = pd.DataFrame(
            np.zeros( ( 1,  part4.shape[1])) * np.nan ,
            columns = part4.columns
            )
        
        taildf[ model.modelname ] = int( model.tailtyp )
        taildf['dpn2'] = taillentmp
        taildf['dpn'] = taillentmp
        
        part4 = taildf.append( part4.append( taildf))
        
        
        
        
        
        
        # create binning from partition
        binntmp = partdf.merge( part4, how= 'left', on='part')
        binntmp[ model.modelname ] = binntmp[ model.modelname ].fillna( method='ffill')
        binntmp['type'] = binntmp[ model.modelname ]
        
        if binntmp.dpn2.sum() != binntmp.shape[0]:
            print('This is not good! binning.dpn2.sum() != binning.shape[0]')
            raise 
            
        taildf = pd.DataFrame(
            np.zeros( ( 1,  binntmp.shape[1])) * np.nan ,
            columns = binntmp.columns
            )
        
        taildf['type'] = int( model.tailtyp )
        taildf['part'] = - 2 * (polid+1) 
        binntmp = binntmp.append( taildf)

        taildf['part'] = - 2 * (polid+1) +1
        binntmp = taildf.append( binntmp )
            
            
        
        
        # map of bs
        bsmaptmp = part4[[ model.modelname ,'dpn2','chromStart']]
        bsmaptmp.columns = ['type','bsbatch','chromStart']
        
        
        
        ###
        binntmp['polyid'] = polid
        binning = binning.append( binntmp)
        
        ###
        bsmaptmp['polyid'] = polid
        bsmaptmp['monocumsum'] = bsmaptmp['bsbatch'].cumsum()
        bsmap = bsmap.append( bsmaptmp)

        ###        
        polid = polid + 1
        taillen += [taillentmp]
        npol += [npoltmp]
    
    
    
    return npol, taillen, binning, bsmap












def genHMM_bin_frac2( model, strpart = 'hicpart', bpmin = 90, fracd=None ):

    ## process chromHMM file
    chrHMM = pd.read_csv( env.strGenetics + '/wgEncodeBroadHmmHuvecHMM.bed.gz', sep='\t', header=None, compression='gzip')
    header = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']
    chrHMM.columns = header[:len(chrHMM.columns)]
    chromHMM_map = pd.read_excel( env.strHome + '/' + model.modelods + '_model.ods', engine="odf", sheet_name= model.modelods_modelsheet, index_col=False)
    
    chrHMM2 = chrHMM.merge( chromHMM_map, how='left', left_on='name', right_on='name' )
    
   
    chrHMM2['dchrbp'] = chrHMM2['chromEnd'] - chrHMM2['chromStart']
    chrHMM2 = chrHMM2[ chrHMM2['dchrbp'] >= bpmin ]
    
    
    ## process gene annotation file
    # hg19_geneannot = pd.read_csv( env.strGenetics + 'gene_annotation.bed', sep='\t|:', index_col=False, header=None)
    # hg19_geneannot.columns = ['chr','Start','End','Type','Name','State','Value','Strand']

    
    
    ##
    try:
        chrL = list(zip( model.chrname, model.chrstart, model.chrend))
    except:
        chrL = [ (model.chrname, model.chrstart, model.chrend) ]
        
    ##
    polid, npol, taillen = 0, [], []
    binning, bsmap = pd.DataFrame([]), pd.DataFrame([])
    for chrname, chrstart, chrend in chrL:
        print( 'Polymer: ', chrname, chrstart, chrend)

        # this selects the region 
        chrsub = chrHMM2[ (chrHMM2['chrom']== chrname) & ( chrHMM2['chromStart'] >= chrstart) & ( chrHMM2['chromEnd'] <= chrend) ]
        # chrsub['chromCenter'] = ((chrsub['chromEnd']+chrsub['chromStart'])/2.).astype(int)

        
        
        # create partition
        if strpart == 'natural':
            part = np.arange( chrsub['chromStart'].values[0], chrsub['chromEnd'].values[-1], model.bpres )
        elif strpart == 'hicpart':
            # map into the hic matrix
            part = np.int_( np.arange( 
                np.round( chrsub['chromStart'].values[0] / model.bpres ) * model.bpres , 
                np.round( chrsub['chromEnd'].values[-1] / model.bpres ) * model.bpres +1, 
                model.bpres
                ) )
        elif strpart == 'given':
            part = partgiven
            
        # pdb.set_trace()
        partdf = pd.DataFrame( part, columns=['part'])
        # partdf['partCenter'] = ((partdf + partdf.shift(-1))/2.)
        partdf.dropna(how='any', axis=0, inplace=True)
        # partdf.partCenter = partdf.partCenter.astype(int)
        # partdf['partCenter'] = ((partdf['chromEnd']+partdf['chromStart'])/2.).astype(int)
        
        # merge as of
        # pdb.set_trace()
        part2 = pd.merge_asof( chrsub[['chromStart','name', model.modelname ]], partdf, 
                              left_on = 'chromStart', right_on ='part' ,  
                              tolerance= model.bpres, direction='nearest' )
        
        part2['partn'] = (part2['part'] / model.bpres)# .astype( np.int32)
        part2[ model.modelname ] = part2[ model.modelname ].astype('int')
        
        # remove classes that are contiguous and identical (this can happen because we remove intervals below threshold bpmin )
        part2['dclass']= part2.shift(-1)[ model.modelname ] - part2[ model.modelname ]
        part4 = part2[ part2['dclass'] != 0]        
        
        # pdb.set_trace()
        # if you want to remove a fraction
        if fracd != None:
            
            # create binning from partition (1)
            binntmp = partdf.merge( part4, how= 'left', on='part')
            binntmp[ model.modelname ] = binntmp[ model.modelname ].fillna( method='ffill')
            binntmp[ 'name' ] = binntmp[ 'name' ].fillna( method='ffill') ##
            binntmp['type'] = binntmp[ model.modelname ]        
            for fracdi in fracd:
                idrep = binntmp[ np.isin( binntmp.type, fracdi['type']) ].sample(frac=fracdi['frac'], random_state=fracdi['rseed']).index
                binntmp.type.loc[ idrep] = fracdi['typenew'] 
                binntmp.name.loc[ idrep] = fracdi['newname'] 
            #
            # pdb.set_trace()
            ## part2 = binntmp.dropna(axis=0, subset=['name'] )
            binntmp['dclass']= binntmp.shift(-1)[ 'type' ] - binntmp[ 'type' ] ##
            part4 = binntmp[ binntmp['dclass'] != 0] ##
            #             
            part4[ model.modelname ] = part4.type ##
            part4.drop('type', 1, inplace=True) ##
            #
            part4['partn'] = (part4['part'] / model.bpres)# .astype( np.int32)
            part4[ model.modelname ] = part4[ model.modelname ].astype('int')

            #             
            ## part2[ model.modelname ] = part2.type
            ## part2.drop('type', 1, inplace=True)
            #
            ## part2['partn'] = (part2['part'] / model.bpres)# .astype( np.int32)
            ## part2[ model.modelname ] = part2[ model.modelname ].astype('int')


        
        # remove classes that are contiguous and identical (this can happen because we remove intervals below threshold bpmin )
        ## part2['dclass']= part2.shift(-1)[ model.modelname ] - part2[ model.modelname ]
        ## part4 = part2[ part2['dclass'] != 0]
        
        # calculate how many beads are in each interval
        part4['dpn'] = part4.shift(-1)['partn'] - part4['partn']
        part4['dpn2'] = part4['dpn']
        
        # deal with last interval
        part4['dpn2'].iloc[-1] = np.round( (chrsub['chromEnd'].values[-1] - part4['chromStart'].values[-1]) / model.bpres )
        part4['dpn'].iloc[-1] = part4['dpn2'].iloc[-1]
        
        # assign 1 bead to all intervals smaller than bpres
        part4.replace({'dpn2':{0:1}}, inplace=True)
        part4['dpn2'] = part4['dpn2'].astype('int')
        
        
        # calculate how many beads are there
        part4['cumpos'] = part4['dpn2'].cumsum() # .shift(+1, fill_value=0) # part4.partn2.values - part4 ['partn2'].iloc[0]
        bslen = part4['dpn2'].sum()
        # estimate tails of polymer
        taillentmp = np.int_( np.round( bslen ** (1/2) ) )
        npoltmp = 2 * taillentmp + bslen
        
        
        # create binning from partition
        taildf = pd.DataFrame(
            np.zeros( ( 1,  part4.shape[1])) * np.nan ,
            columns = part4.columns
            )
        
        taildf[ model.modelname ] = int( model.tailtyp )
        taildf['dpn2'] = taillentmp
        taildf['dpn'] = taillentmp
        
        part4 = taildf.append( part4.append( taildf))
        
        
        
        
        
        
        # create binning from partition
        binntmp = partdf.merge( part4, how= 'left', on='part')
        binntmp[ model.modelname ] = binntmp[ model.modelname ].fillna( method='ffill')
        binntmp['type'] = binntmp[ model.modelname ]
        pdb.set_trace()
        if binntmp.dpn2.sum() != binntmp.shape[0]:
            print('This is not good! binning.dpn2.sum() != binning.shape[0]')
            raise 
            
        taildf = pd.DataFrame(
            np.zeros( ( 1,  binntmp.shape[1])) * np.nan ,
            columns = binntmp.columns
            )
        
        taildf['type'] = int( model.tailtyp )
        taildf['part'] = - 2 * (polid+1) 
        binntmp = binntmp.append( taildf)

        taildf['part'] = - 2 * (polid+1) +1
        binntmp = taildf.append( binntmp )
            
            
        
        
        # map of bs
        bsmaptmp = part4[[ model.modelname ,'dpn2','chromStart']]
        bsmaptmp.columns = ['type','bsbatch','chromStart']
        
        
        
        ###
        binntmp['polyid'] = polid
        binning = binning.append( binntmp)
        
        ###
        bsmaptmp['polyid'] = polid
        bsmaptmp['monocumsum'] = bsmaptmp['bsbatch'].cumsum()
        bsmap = bsmap.append( bsmaptmp)

        ###        
        polid = polid + 1
        taillen += [taillentmp]
        npol += [npoltmp]
    
    
    
    return npol, taillen, binning, bsmap






def genArtificial( model, strpart= 'natural' ):
    ## process .bed artificial sysstem file
    header = ['chrom', 'chromStart', 'chromEnd', 'name', 'annot', 'strand', 'score' ]
    chrArt = pd.read_excel( model.artifSys[0] , engine="odf", sheet_name= model.artifSys[1], index_col=False, usecols=header)
    chrArt.columns = header[:len(chrArt.columns)]
    chromArt_map = pd.read_excel( env.strHome + '/' + model.modelods + '_model.ods', engine="odf", sheet_name= model.modelods_modelsheet, index_col=False)
    
    chromArt2 = chrArt.merge( chromArt_map, how='left', left_on='name', right_on='name' )
   
    chromArt2['dchrbp'] = chromArt2['chromEnd'] - chromArt2['chromStart']
    
    
    parampd = pd.read_excel( model.artifSys[0] , engine="odf", sheet_name= 'param' , index_col=False)
    model.bpres = parampd[parampd.ssystem == model.artifSys[1] ].bpres.values[0]
    taillentmp = parampd[parampd.ssystem == model.artifSys[1] ].taillength.values[0]
            
    chromArt2['chromStart'] = chromArt2['chromStart'] * model.bpres
    chromArt2['chromEnd'] = chromArt2['chromEnd'] * model.bpres
    chrNamAnnot = chromArt2[['name','annot']].drop_duplicates()
    chrNamAnnot['nameAnnot'] = chrNamAnnot['name'] + '+' + chrNamAnnot['annot']
    chrNamAnnot['class'] = chrNamAnnot.index

    chromArt2 = chromArt2.merge( chrNamAnnot, on=['name','annot'], how='left')


    ##
    chrnames = chromArt2.chrom.unique()
    chrL = []
    for chrnamei in chrnames:
        chrL += [[
            chrnamei ,
            chromArt2[ chromArt2.chrom == chrnamei ].chromStart.iloc[0] ,
            chromArt2[ chromArt2.chrom == chrnamei ].chromEnd.iloc[-1]
            ]]
    

    ##
    polid, npol, taillen = 0, [], []
    binning, bsmap = pd.DataFrame([]), pd.DataFrame([])
    for chrname, chrstart, chrend in chrL:
        print( 'Polymer: ', chrname, chrstart, chrend)

        # this selects the region 
        chrsub = chromArt2[ (chromArt2['chrom']== chrname) & ( chromArt2['chromStart'] >= chrstart) & ( chromArt2['chromEnd'] <= chrend) ]
        # chrsub['chromCenter'] = ((chrsub['chromEnd']+chrsub['chromStart'])/2.).astype(int)

        
        
        # create partition
        if strpart == 'natural':
            part = np.arange( chrsub['chromStart'].values[0], chrsub['chromEnd'].values[-1], model.bpres )
        elif strpart == 'hicpart':
            # map into the hic matrix
            part = np.int_( np.arange( 
                np.round( chrsub['chromStart'].values[0] / model.bpres ) * model.bpres , 
                np.round( chrsub['chromEnd'].values[-1] / model.bpres ) * model.bpres +1, 
                model.bpres
                ) )
        elif strpart == 'given':
            part = partgiven
            
        # pdb.set_trace()
        partdf = pd.DataFrame( part, columns=['part'])
        # partdf['partCenter'] = ((partdf + partdf.shift(-1))/2.)
        partdf.dropna(how='any', axis=0, inplace=True)
        # partdf.partCenter = partdf.partCenter.astype(int)
        # partdf['partCenter'] = ((partdf['chromEnd']+partdf['chromStart'])/2.).astype(int)
        
        # merge as of
        # pdb.set_trace()
        part2 = pd.merge_asof( chrsub[['chromStart','name', 'annot', 'strand', 'class' , model.modelname]], partdf, 
                              left_on = 'chromStart', right_on ='part' ,  
                              tolerance= model.bpres, direction='nearest' )
        
        part2['partn'] = (part2['part'] / model.bpres) # .astype( np.int32)
        part2[ 'class' ] = part2[ 'class' ].astype('int')
        
        # remove classes that are contiguous and identical (this can happen because we remove intervals below threshold bpmin )
        part2['dclass']= part2.shift(-1)[ 'class' ] - part2[ 'class' ]
        part4 = part2[ part2['dclass'] != 0]        
        
        
        # calculate how many beads are in each interval
        part4['dpn'] = part4.shift(-1)['partn'] - part4['partn']
        part4['dpn2'] = part4['dpn']
        
        # deal with last interval
        part4['dpn2'].iloc[-1] = np.round( (chrsub['chromEnd'].values[-1] - part4['chromStart'].values[-1]) / model.bpres )
        part4['dpn'].iloc[-1] = part4['dpn2'].iloc[-1]
        
        # assign 1 bead to all intervals smaller than bpres
        part4.replace({'dpn2':{0:1}}, inplace=True)
        part4['dpn2'] = part4['dpn2'].astype('int')
        
        
        # calculate how many beads are there
        part4['cumpos'] = part4['dpn2'].cumsum() # .shift(+1, fill_value=0) # part4.partn2.values - part4 ['partn2'].iloc[0]
        bslen = part4['dpn2'].sum()
        # estimate tails of polymer
        # taillentmp = np.int_( np.round( bslen ** (1/2) ) )
        npoltmp = 2 * taillentmp + bslen
        
        
        # tails
        taildf = pd.DataFrame(
            np.zeros( ( 1,  part4.shape[1])) * np.nan ,
            columns = part4.columns
            )
        
        taildf[ model.modelname ] = int( model.tailtyp )
        taildf['dpn2'] = taillentmp
        taildf['dpn'] = taillentmp
        
        part4 = taildf.append( part4.append( taildf))
        
        
        
        
        
        
        # create binning from partition
        binntmp = partdf.merge( part4, how= 'left', on='part')
        binntmp[ 'class' ] = binntmp[ 'class' ].fillna( method='ffill')
        binntmp['type'] = binntmp[ model.modelname ]

        if binntmp.dpn2.sum() != binntmp.shape[0]:
            print('This is not good! binning.dpn2.sum() != binning.shape[0]')
            raise 
            
        taildf = pd.DataFrame(
            np.zeros( ( 1,  binntmp.shape[1])) * np.nan ,
            columns = binntmp.columns
            )
        # pdb.set_trace()
        taildf['type'] = int( model.tailtyp )
        taildf['part'] = - 2 * (polid+1) 
        binntmp = binntmp.append( taildf)

        taildf['part'] = - 2 * (polid+1) +1
        binntmp = taildf.append( binntmp )
            
            
        
        
        # map of bs
        bsmaptmp = part4[[ model.modelname ,'dpn2','chromStart']]
        bsmaptmp.columns = ['type','bsbatch','chromStart']
        
        
        
        ###
        binntmp['polyid'] = polid
        binning = binning.append( binntmp)
        
        ###
        bsmaptmp['polyid'] = polid
        bsmaptmp['monocumsum'] = bsmaptmp['bsbatch'].cumsum()
        bsmap = bsmap.append( bsmaptmp)

        ###        
        polid = polid + 1
        taillen += [taillentmp]
        npol += [npoltmp]
        
        
        ### genome
        genome = [ int(bsmap.chromStart.min())]
        genome += [ int(bsmap.chromStart.max() + model.bpres)]
        genome += [ model.bpres]
    
    
    
    return npol, taillen, binning, bsmap, genome





def genReal( model, strpart= 'natural' ):
    ## process .bed artificial sysstem file
    header = ['chrom', 'chromStart', 'chromEnd', 'name', 'annot', 'strand', 'score', 'chromStartBp', 'chromEndBp' ]
    chrArt = pd.read_excel( model.artifSys[0] , engine="odf", sheet_name= model.artifSys[1], index_col=False, usecols=header)
    chrArt.columns = header[:len(chrArt.columns)]
    chromArt_map = pd.read_excel( env.strHome + '/' + model.modelods + '_model.ods', engine="odf", sheet_name= model.modelods_modelsheet, index_col=False)
    
    chromArt2 = chrArt.merge( chromArt_map, how='left', left_on='name', right_on='name' )
   
    chromArt2['dchrbp'] = chromArt2['chromEnd'] - chromArt2['chromStart']
    
    
    parampd = pd.read_excel( model.artifSys[0] , engine="odf", sheet_name= 'param' , index_col=False)
    model.bpres = parampd[parampd.ssystem == model.artifSys[1] ].bpres.values[0]
    taillentmp = parampd[parampd.ssystem == model.artifSys[1] ].taillength.values[0]
            
    chromArt2['chromStart'] = chromArt2['chromStart'] * model.bpres
    chromArt2['chromEnd'] = chromArt2['chromEnd'] * model.bpres
    chrNamAnnot = chromArt2[['name','annot']].drop_duplicates()
    chrNamAnnot['nameAnnot'] = chrNamAnnot['name'] + '+' + chrNamAnnot['annot']
    chrNamAnnot['class'] = chrNamAnnot.index

    chromArt2 = chromArt2.merge( chrNamAnnot, on=['name','annot'], how='left')


    ##
    chrnames = chromArt2.chrom.unique()
    chrL = []
    for chrnamei in chrnames:
        chrL += [[
            chrnamei ,
            chromArt2[ chromArt2.chrom == chrnamei ].chromStart.iloc[0] ,
            chromArt2[ chromArt2.chrom == chrnamei ].chromEnd.iloc[-1]
            ]]
    

    ##
    polid, npol, taillen = 0, [], []
    binning, bsmap = pd.DataFrame([]), pd.DataFrame([])
    for chrname, chrstart, chrend in chrL:
        print( 'Polymer: ', chrname, chrstart, chrend)

        # this selects the region 
        chrsub = chromArt2[ (chromArt2['chrom']== chrname) & ( chromArt2['chromStart'] >= chrstart) & ( chromArt2['chromEnd'] <= chrend) ]
        # chrsub['chromCenter'] = ((chrsub['chromEnd']+chrsub['chromStart'])/2.).astype(int)

        
        
        # create partition
        if strpart == 'natural':
            part = np.arange( chrsub['chromStart'].values[0], chrsub['chromEnd'].values[-1], model.bpres )
        elif strpart == 'hicpart':
            # map into the hic matrix
            part = np.int_( np.arange( 
                np.round( chrsub['chromStart'].values[0] / model.bpres ) * model.bpres , 
                np.round( chrsub['chromEnd'].values[-1] / model.bpres ) * model.bpres +1, 
                model.bpres
                ) )
        elif strpart == 'given':
            part = partgiven
            
        # pdb.set_trace()
        partdf = pd.DataFrame( data={
            'part' : part})
        # partdf['partCenter'] = ((partdf + partdf.shift(-1))/2.)
        partdf.dropna(how='any', axis=0, inplace=True)
        # partdf.partCenter = partdf.partCenter.astype(int)
        # partdf['partCenter'] = ((partdf['chromEnd']+partdf['chromStart'])/2.).astype(int)
        
        # merge as of
        # pdb.set_trace()
        part2 = pd.merge_asof( chrsub[['chromStart','name', 'annot', 'strand', 'class', 'chromStartBp','chromEndBp' , model.modelname]], 
                              partdf, 
                              left_on = 'chromStart', right_on ='part' ,  
                              tolerance= model.bpres, direction='nearest' )
        
        part2['partn'] = (part2['part'] / model.bpres) # .astype( np.int32)
        part2[ 'class' ] = part2[ 'class' ].astype('int')
        
        # remove classes that are contiguous and identical (this can happen because we remove intervals below threshold bpmin )
        # part2['dclass']= part2.shift(-1)[ 'class' ] - part2[ 'class' ]
        part2['dclass']= part2[ 'class' ] - part2[ 'class' ].shift(1)
        part4 = part2[ part2['dclass'] != 0]        
        
        
        # calculate how many beads are in each interval
        part4['dpn'] = part4.shift(-1)['partn'] - part4['partn']
        part4['dpn2'] = part4['dpn']
        
        # deal with last interval
        part4['dpn2'].iloc[-1] = np.round( (chrsub['chromEnd'].values[-1] - part4['chromStart'].values[-1]) / model.bpres )
        part4['dpn'].iloc[-1] = part4['dpn2'].iloc[-1]
        
        # assign 1 bead to all intervals smaller than bpres
        part4.replace({'dpn2':{0:1}}, inplace=True)
        part4['dpn2'] = part4['dpn2'].astype('int')
        
        
        # calculate how many beads are there
        part4['cumpos'] = part4['dpn2'].cumsum() # .shift(+1, fill_value=0) # part4.partn2.values - part4 ['partn2'].iloc[0]
        bslen = part4['dpn2'].sum()
        # estimate tails of polymer
        # taillentmp = np.int_( np.round( bslen ** (1/2) ) )
        npoltmp = 2 * taillentmp + bslen
        
        
        # tails
        taildf = pd.DataFrame(
            np.zeros( ( 1,  part4.shape[1])) * np.nan ,
            columns = part4.columns
            )
        
        taildf[ model.modelname ] = int( model.tailtyp )
        taildf['dpn2'] = taillentmp
        taildf['dpn'] = taillentmp
        
        part4 = taildf.append( part4.append( taildf))
        
        
        
        
        
        # =============================================================================
        #         # create binning from partition
        # =============================================================================
        binntmp = partdf.merge( part4, how= 'left', on='part')
        pdb.set_trace()
        
        binntmp['chromStartBpD2'] = binntmp['chromStartBp'] - binntmp['chromStartBp'].shift(1)
        binntmp['chromStartBpD1'] = binntmp.chromStartBp - binntmp.chromStartBp + model.bpres
        binntmp['chromStartBpD'] = binntmp.chromStartBp - binntmp.chromStartBp
        binntmp[ 'chromStartBpD' ] = binntmp[ 'chromStartBpD' ].fillna( value=model.bpres)
        binntmp[ 'chromStartBp2' ] = (binntmp[ 'chromStartBpD' ]+ binntmp['chromStartBpD1'] + binntmp['chromStartBpD2'].fillna(value=0)).cumsum() + binntmp[ 'chromStartBp' ].iloc[0] 
        
        binntmp[ 'class' ] = binntmp[ 'class' ].fillna( method='ffill')
        binntmp['type'] = binntmp[ model.modelname ]

        if binntmp.dpn2.sum() != binntmp.shape[0]:
            print('This is not good! binning.dpn2.sum() != binning.shape[0]')
            raise 
            
        taildf = pd.DataFrame(
            np.zeros( ( 1,  binntmp.shape[1])) * np.nan ,
            columns = binntmp.columns
            )
        # pdb.set_trace()
        taildf['type'] = int( model.tailtyp )
        taildf['part'] = - 2 * (polid+1) 
        binntmp = binntmp.append( taildf)

        taildf['part'] = - 2 * (polid+1) +1
        binntmp = taildf.append( binntmp )
            
            
        
        
        # map of bs
        bsmaptmp = part4[[ model.modelname ,'dpn2','chromStart']]
        bsmaptmp.columns = ['type','bsbatch','chromStart']
        
        
        
        ###
        binntmp['polyid'] = polid
        binning = binning.append( binntmp)
        
        ###
        bsmaptmp['polyid'] = polid
        bsmaptmp['monocumsum'] = bsmaptmp['bsbatch'].cumsum()
        bsmap = bsmap.append( bsmaptmp)

        ###        
        polid = polid + 1
        taillen += [taillentmp]
        npol += [npoltmp]
        
        
        ### genome
        genome = [ int(bsmap.chromStart.min())]
        genome += [ int(bsmap.chromStart.max() + model.bpres)]
        genome += [ model.bpres]
    
    
    
    return npol, taillen, binning, bsmap, genome




















def defLEtypes( model, strmode = 'sameAsBind_randOrient_stalltails', rseed=42, coheStallSitesFract=None ):

    if strmode == 'sameAsBind_randOrient_stalltails':
        
        # same as binding types
        model.bsmap['monocumsum2'] = model.bsmap['bsbatch'].cumsum().shift(fill_value=0)
        bsmapstall = model.bsmap[ np.isin( model.bsmap.type, model.idlopestall )]

        # remove a fraction of sites because not real insulation sites    
        if coheStallSitesFract is not None:
            bsmapstall = bsmapstall.sample( frac=coheStallSitesFract, random_state=rseed)

        # add random ctcf orientation    
        np.random.seed( rseed)
        bsmapstall['stallOrientation'] = np.random.randint(0,2,bsmapstall.shape[0])
    
        idd1not = []
        for stallid in range(0,2):
            bstmp = bsmapstall[ bsmapstall['stallOrientation']== stallid]
    
            idd1not += [ np.concatenate(
                [ np.arange( int(rrow.monocumsum2), int(rrow.monocumsum)) for idx, rrow in bstmp.iterrows()]
                ).tolist()    ]
            

        # add stalling on tails
        for stallid in range(0,2):
                idd1not[ stallid] += list(range(0,model.taillen[0]))
                idd1not[ stallid] += list(range( model.npol[0]-model.taillen[0], model.npol[0] ))
                
    
    return idd1not





def defLEtypes2( ssystem, model, strmode = 'sameAsBind_randOrient_stalltails', rseed=42, coheStallSitesFract=None, strSys='espresso' ):

    if strmode == 'sameAsBind_randOrient_stalltails':
        random.seed( rseed)
        if strSys =='espresso':
            idstall = ssystem.part[:].id[ np.isin( ssystem.part[:].type, model.idlopestall )]
        elif strSys == 'df':
            idstall = ssystem.Index[ np.isin( ssystem.type, model.idlopestall )].values
            
        # remove a fraction of sites because not real insulation sites    
        idstall = np.array( random.sample( idstall.tolist(), int(idstall.size * coheStallSitesFract) ) )

        # add random ctcf orientation    
        np.random.seed( rseed)
        idstallbool = np.random.randint(0,2,idstall.size)
        idd1not = [
            list(idstall[ idstallbool==0]),
            list(idstall[ idstallbool==1])
            ]
        
        
        # add stalling on tails
        for stallid in range(0,2):
            # idd1not[ stallid] += list(range(0,model.taillen[0]))
            # idd1not[ stallid] += list(range( model.npol[0]-model.taillen[0], model.npol[0] ))
            if strSys =='espresso':
                idd1not[ stallid] += list( ssystem.part[:].id[ np.isin( ssystem.part[:].type, model.stallAnywayTypeLope)] )
            elif strSys == 'df':
                idd1not[ stallid] += list( ssystem.Index[ np.isin( ssystem.type, model.stallAnywayTypeLope)] )
            
                
            
    elif strmode in ['artOrient_stalltails','artOrient_stalltails_invertOrient']:
        if strSys =='espresso':
            idlopestallnotail = list(set(model.idlopestall)-set([model.tailtyp]))
            idstall = ssystem.part[:].id[ np.isin( ssystem.part[:].type, idlopestallnotail )]
        elif strSys == 'df':
            idlopestallnotail = list(set(model.idlopestall)-set([model.tailtyp]))
            idstall = ssystem.Index[ np.isin( ssystem.type, idlopestallnotail )].values
                    
        # add ctcf from model
        strandpd = model.binnffill[ 
            ( np.isin( model.binnffill.strand, ['+','-']) ) & 
            ( np.isin( model.binnffill.type, idlopestallnotail ) )
            ][['Index','strand']]

        if strmode in ['artOrient_stalltails']:
            idd1not = [
                list(idstall[ strandpd.strand=='+']),
                list(idstall[ strandpd.strand=='-'])
                ]        
        elif strmode in ['artOrient_stalltails_invertOrient']:
            idd1not = [
                list(idstall[ strandpd.strand=='-']),
                list(idstall[ strandpd.strand=='+'])
                ]        

        
        # add stalling on tails
        for stallid in range(0,2):
            # idd1not[ stallid] += list(range(0,model.taillen[0]))
            # idd1not[ stallid] += list(range( model.npol[0]-model.taillen[0], model.npol[0] ))
            if strSys =='espresso':
                idd1not[ stallid] += list( ssystem.part[:].id[ np.isin( ssystem.part[:].type, model.stallAnywayTypeLope)] )
            elif strSys == 'df':
                idd1not[ stallid] += list( ssystem.Index[ np.isin( ssystem.type, model.stallAnywayTypeLope)] )
                        
    
    return idd1not




def defReeltypes( model, strmode = 'sameAsBind_randOrient_stalleverywhere', rseed=42 ):

    if strmode == 'sameAsBind_randOrient_stalleverywhere':
        
        # same as binding types
        model.bsmap['monocumsum2'] = model.bsmap['bsbatch'].cumsum().shift(fill_value=0)
        bsmapstall = model.bsmap[ np.isin( model.bsmap.type, model.idreelstall )]
    
        # add random ctcf orientation    
        np.random.seed( rseed)
        bsmapstall['stallOrientation'] = np.random.randint(0,2,bsmapstall.shape[0])
    
        idd1not = []
        for stallid in range(0,2):
            bstmp = bsmapstall[ bsmapstall['stallOrientation']== stallid]
    
            idd1not += [ np.concatenate(
                [ np.arange( int(rrow.monocumsum2), int(rrow.monocumsum)) for idx, rrow in bstmp.iterrows()]
                ).tolist()    ]
            
        # add stalling on tails
        for stallid in range(0,2):
            if stallid == 0:
                idd1not[ stallid] += list(range(0,model.taillen[0]))
            elif stallid == 1:
                idd1not[ stallid] += list(range( model.npol[0]-model.taillen[0], model.npol[0] ))
                
    
    return idd1not















def hicBrackley2_bin( bpres, chrname, chrstart, chrend, bpmin = 90 ):

    chrHMM = pd.read_csv( strGenetics + '/wgEncodeBroadHmmHuvecHMM.bed.gz', sep='\t', header=None, compression='gzip')
    header = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']
    chrHMM.columns = header[:len(chrHMM.columns)]
    chromHMM_map = pd.read_excel( strHome + '/modelBloom2021.ods', engine="odf", sheet_name='encodeHMM', index_col=False)

    chrHMM2 = chrHMM.merge( chromHMM_map, how='left', left_on='name', right_on='name' )
    
    # check mergin went good
    chrHMM2.class_id.unique()
    
    # check population distribution of classes
    chrHMM2.groupby('class_id').count()
    chrHMM2.groupby('brackley2016').count()
    
    
    chrHMM2['dchrbp'] = chrHMM2['chromEnd'] - chrHMM2['chromStart']
    chrHMM2 = chrHMM2[ chrHMM2['dchrbp'] >= bpmin ]
    
    
    # this selects the region # mdify to add >= and <= !!!    
    chrsub = chrHMM2[ (chrHMM2['chrom']== chrname) & ( chrHMM2['chromStart'] > chrstart) & ( chrHMM2['chromEnd'] < chrend) ]
    
    
    part = np.arange( chrsub['chromStart'].values[0], chrsub['chromEnd'].values[-1], bpres )
    partdf = pd.DataFrame( part, columns=['part'])
    
    
    # part2 = pd.merge_asof( chrsub[['chromStart','name','brackley2016']], partdf, left_on = 'chromStart', right_on ='part' ,  tolerance= bpres, direction='nearest' )
    part2 = pd.merge_asof( chrsub[['chromStart','name','brackley2016']], partdf, left_on = 'chromStart', right_on ='part' ,  tolerance= bpres )
    part2['partn'] = (part2['part'] / bpres)# .astype( np.int32)
    
    
    # remove classes that are contiguous and identical (this can happen because we remove intervals below threshold bpmin )
    part2['dclass']= part2.shift(-1)['brackley2016'] - part2['brackley2016']
    part4 = part2[ part2['dclass'] != 0]
    
    # calculate how many beads are in each interval
    part4['dpn'] = part4.shift(-1)['partn'] - part4['partn']
    part4['dpn2'] = part4['dpn']
    
    # deal with last interval
    part4['dpn2'].iloc[-1] = np.round( (chrsub['chromEnd'].values[-1] - part4['chromStart'].values[-1]) / bpres )
    
    # assign 1 bead to all intervals smaller than bpres
    part4.replace({'dpn2':{0:1}}, inplace=True)
    part4['dpn2'] = part4['dpn2'].astype('int')
    
    
    # checks
    # =============================================================================
    #     part4['dpn'].sum() * bpres
    #     part4['dpn2'].sum() * bpres
    #     part4['dpn2'].sum()
    # =============================================================================
    
    
    part4['partn3'] = part4['dpn2'].cumsum().shift(+1, fill_value=0) # part4.partn2.values - part4 ['partn2'].iloc[0]
    
    bs = np.zeros( ( len( part4.brackley2016.unique()) , part4['dpn2'].sum() ) ) * np.nan
    for idx, parti in part4.reset_index().iterrows() :
        bstmp = np.arange( parti.partn3, parti.partn3 + parti.dpn2 )
        bs[ parti.brackley2016, bstmp ] = bstmp
    
    
    part4['dstart']= part4.shift(-1)['chromStart'] - part4['chromStart']
    part4['dstartn']= part4['dstart'] / bpres
    
    part4[['chromStart','brackley2016','partn3','dpn2','dstartn']]
    
    
    binding = []
    for bsi in range( bs.shape[0]-1 ):
        bstmp = bs[ bsi, :]
        binding += [list( bstmp[ ~np.isnan( bstmp)] + np.int_( bs.shape[1] ** (1/2) ) )]    

    # =============================================================================
    # # old wrong way
    # npol = 2 * np.int_( bs.shape[1] ** (1/2) ) + bs.shape[1]
    # tail = np.int_( bs.shape[1] ** (1/2) )
    # =============================================================================
    # correct way
    npol = 2 * np.int_( np.round( bs.shape[1] ** (1/2) )) + bs.shape[1]
    tail = np.int_( np.round( bs.shape[1] ** (1/2) ) )

    part5 =part4[['chromStart','part','dpn','partn3']]
    binning = partdf.merge( part5, how= 'left', on='part')


    # map into the hic matrix
    ppart = np.int_( np.arange( np.round( chrsub['chromStart'].values[0] / bpres)* bpres, chrsub['chromEnd'].values[-1], bpres) )
    ppartdf = pd.DataFrame( ppart, columns=['ppart'])

    partpart = pd.concat( (ppartdf, partdf),1)
    
    binning['beadPos'] = binning.apply( binningwhere, 1)
    binning = binning.merge( partpart, how='inner', on='part')


    
    ppart2 = pd.merge_asof( binning[['beadPos','chromStart']], binning[['part','ppart']], 
                           left_on = 'beadPos', right_on ='ppart' ,  
                           tolerance= bpres, direction='nearest' )
    
    # ppart2 = pd.merge_asof( chrsub[['chromStart','name','brackley2016']], binning[['part2','part','ppart']], left_on = 'chromStart', right_on ='ppart' ,  tolerance= bpres, direction='nearest' )
   
    # ppart5 =ppart4[['chromStart','part2','ppart','part']]

    # bbinning = partdf.merge( ppart5, how= 'left', on='part')
    # bbinning = binning.merge( ppart2[['part2','chromStart']], how= 'left', on=['part2','chromStart'])
    bbinning = ppart2[['part','chromStart','ppart','beadPos']]
    bbinning.columns = ['part1','chromStart','part','beadPos']
    
    return binding, npol, binning, bbinning, tail






def binningwhere( x):
    if np.isnan( x.chromStart ):
        return np.int_( x.part )
    else:
        return np.int_( x.chromStart )






def addTailCons( bpres, chrname, chrstart, chrend, bpmin = 90 ):
    N = 10
    ctail = ['m']

    binding, npol = genBrackley2( bpres, chrname, chrstart, chrend, bpmin = 90 )
    binding += [ list( range(0, N )) + list( range( npol-1 , npol-1 + N )) ]
    npol = npol + 2 * N
    
    return binding, npol, ctail













def genBrackley( bpres, chrname, Npart, chrstart ):
    chrHMM = pd.read_csv( strGenetics + '/wgEncodeBroadHmmHuvecHMM.bed.gz', sep='\t', header=None, compression='gzip')
    header = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']
    chrHMM.columns = header[:len(chrHMM.columns)]
    
    # chrHMM[ chrHMM['chrom']=='chr7'].head()
    
    strGenTable = 'read'
    if strGenTable == 'build':
        chrHMM.name.unique()
        chromHMM_map = chrHMM.name.str.split('_', n=1, expand=True).drop_duplicates()
        chromHMM_map.columns = ['class_n','class_id']
        chromHMM_map['name'] = chrHMM.name.drop_duplicates()
        chromHMM_map.to_csv( strGeneticsHome + '/broadChromHMM_table.csv', sep='\t', index=False )
        
    elif strGenTable == 'read':
        chromHMM_map = pd.read_csv( strGeneticsHome + '/broadChromHMM_table.csv', sep='\t', index_col=False)
        # chromHMM_map.columns = ['category','color','definition','type']
    
    chrHMM2 = chrHMM.merge( chromHMM_map, how='left', left_on='name', right_on='name' )
    
    # check mergin went good
    chrHMM2.class_id.unique()
    
    # check population distribution of classes
    chrHMM2.groupby('class_id').count()
    chrHMM2.groupby('brackley2016').count()
    
    
    chrHMM2['dchrbp'] = chrHMM2['chromEnd'] - chrHMM2['chromStart']
    chrHMM2 = chrHMM2[ chrHMM2['dchrbp'] >= 90 ]
    
    
    chr7hmm = chrHMM2[ chrHMM2['chrom']== chrname]
    
    
    part = np.arange( chr7hmm['chromStart'].values[0], chr7hmm['chromEnd'].values[-1], bpres )
    partdf = pd.DataFrame( part, columns=['part'])
    
    
    part2 = pd.merge_asof( chr7hmm[['chromStart','name','brackley2016']], partdf, left_on = 'chromStart', right_on ='part' ,  tolerance= bpres )
    part2['partn'] = (part2['part'] / bpres)# .astype( np.int32)
    
    
    #
    part2['dclass']= part2.shift(-1)['brackley2016'] - part2['brackley2016']
    part3 = part2[ part2['dclass'] != 0]
    
    part3['dpn'] = part3.shift(-1)['partn'] - part3['partn']
    part3['dpn2'] = part3['dpn']
    
    part3['dpn2'].iloc[-1] = np.round( (chr7hmm['chromEnd'].values[-1] - part3['chromStart'].values[-1]) / bpres )
    part3.replace({'dpn2':{0:1}}, inplace=True)
    part3['dpn2'] = part3['dpn2'].astype('int')
    
    
    # checks
    part3['dpn'].sum() * bpres
    part3['dpn2'].sum() * bpres
    
    
    
    
    part3['partn2'] = part3['dpn2'].cumsum()
    part4 = part3[ part3['chromStart'] > chrstart ]
    part4 = part4[ part4 ['partn2'] - part4 ['partn2'].iloc[0] < Npart[-1] ]
    part4['partn3'] = part4.partn2.values - part4 ['partn2'].iloc[0]
    binding = []
    for ptyp in range( part4.brackley2016.unique().size -1):
        parttmp = part4[ part4.brackley2016 == ptyp ]
        binding += [list( parttmp.partn3.values )]
    
    

    return binding









def distM( pol_type, ssystem) :
    # this actually works for whatever combination list of types in pol_type
    ppos = ssystem.part[:].pos
    typv = ssystem.part[:].type
    
# =============================================================================
#     types= set(system.part[:].type)
#     
#     dm = []
#     for typi in types:
#         ppostmp = ppos [ typv == typi, : ]
#     
#         dm = dm + [scy.spatial.distance_matrix( ppostmp, ppostmp )]
#         
# =============================================================================
        
        
    polpos = ppos [ [True if tyi in pol_type else False for tyi in typv], : ]
    poldm = scy.spatial.distance_matrix( polpos, polpos )

    return poldm






def distM2( pol_type, ppos, typv) :
       
    polpos = ppos [ [True if tyi in pol_type else False for tyi in typv], : ]
    poldm = scy.spatial.distance_matrix( polpos, polpos )

    return poldm



def distM3( df, typv) :
    df['index'] = df.index
    dft = df.set_index('type').loc[ typv]
    dft = dft.sort_values( 'index')
    
    
    
    for idx, tme in enumerate( dft.time.unique() ):
        if idx == 0 :
            ppos = dft[ dft.time == tme][['x','y','z']]
            dm = scy.spatial.distance_matrix( ppos, ppos )
        else :
            ppos = dft[ dft.time == tme][['x','y','z']]
            posdm = scy.spatial.distance_matrix( ppos, ppos )

            dm = np.concatenate( (dm, posdm[:,:,None] ), 2)

    return dm



def distM3_4( df, typv) :
    df['index'] = df.index
    dft = df.set_index('type').loc[ typv]
    dft = dft.sort_values( 'index')
    
    
    
    for idx, tme in enumerate( dft.time.unique() ):
        if idx == 0 :
            ppos = dft[ dft.time == tme][['x','y','z']]
            dm = scy.spatial.distance_matrix( ppos, ppos )
        else :
            ppos = dft[ dft.time == tme][['x','y','z']]
            posdm = scy.spatial.distance_matrix( ppos, ppos )

            dm = np.concatenate( (dm, posdm[:,:,None] ), 2)

    # pdb.set_trace()
    dtype = dft[ dft.time == tme].reset_index().type

    return dm, dtype





def distM3_1( df, typv, binning, chrstart, chrend, npol) :
    df['Index'] = df.index
    dft = df.set_index('type').loc[ typv]
    dft = dft.sort_values( 'Index').reset_index()
    
    #
    binning['Index'] = binning.reset_index().index
    ### remove tails
    poltail = removeTails( chrstart, chrend, npol)
    
    
    for idx, tme in enumerate( dft.time.unique() ):
        ppos0 = dft[ dft.time == tme] # [['x','y','z','bin']]
        #
        ppos0 = ppos0.iloc[ poltail :-poltail ].reset_index()
        ppos0['Index'] = ppos0.index
        
        #
        ppos02 = ppos0.merge( binning, on='Index', how='inner')


        ppos = ppos02[['x','y','z','part']].groupby('part').mean().reset_index()[['x','y','z']]

        if idx == 0 :
            dm = scy.spatial.distance_matrix( ppos, ppos )
        else :
            posdm = scy.spatial.distance_matrix( ppos, ppos )

            dm = np.concatenate( (dm, posdm[:,:,None] ), 2)

    return dm






def distM3_1p1( df, typv, binning, chrstart, chrend, npol) :
    df['Index'] = df.index
    dft = df.set_index('type').loc[ typv]
    dft = dft.sort_values( 'Index').reset_index()
    
    #
    binning = binning[ binning.part >= 0]
    binning['Index'] = binning.reset_index().index
    ### remove tails
    poltail = removeTails( chrstart, chrend, npol)
    
    
    for idx, tme in enumerate( dft.time.unique() ):
        ppos0 = dft[ dft.time == tme] # [['x','y','z','bin']]
        #
        ppos0 = ppos0.iloc[ poltail :-poltail ].reset_index()
        ppos0['Index'] = ppos0.index
        
        #
        ppos02 = ppos0.merge( binning, on='Index', how='inner')


        ppos = ppos02[['x','y','z','part']].groupby('part').mean().reset_index()[['x','y','z']]

        if idx == 0 :
            dm = scy.spatial.distance_matrix( ppos, ppos )
        else :
            posdm = scy.spatial.distance_matrix( ppos, ppos )

            dm = np.concatenate( (dm, posdm[:,:,None] ), 2)

    return dm







def distM3_2( df, typv, binning, chrstart, chrend, npol) :
    df['Index'] = df.index
    dft = df.set_index('type').loc[ typv]
    dft = dft.sort_values( 'Index').reset_index()
    
    ### remove tails
    binning['Index'] = binning.reset_index().index
    poltail = removeTails( chrstart, chrend, npol)
    
    
    for idx, tme in enumerate( dft.time.unique() ):
        ppos0 = dft[ dft.time == tme] # [['x','y','z','bin']]
        #
        ppos0 = ppos0.iloc[ poltail :-poltail ].reset_index()
        ppos0['Index'] = ppos0.index
        
        #
        ppos02 = ppos0.merge( binning, on='Index', how='inner', suffixes=['','_r'])


        ppos = ppos02[['x','y','z','part']].groupby('part').mean().reset_index()[['x','y','z']]

        if idx == 0 :
            dm = scy.spatial.distance_matrix( ppos, ppos )
        else :
            posdm = scy.spatial.distance_matrix( ppos, ppos )

            dm = np.concatenate( (dm, posdm[:,:,None] ), 2)


    dftype = ppos02[['type','part']]

    return dm, dftype










def pclCM( df, model, param, mode='pcl'  ):
    
    
    #######
    sys.setrecursionlimit(10000)     
    
    clcutoff = param.clcutoffDMP
    # min_samples = 4
    d0 = param.d0DMP



    def clmolfunc(dmmoli):
        
        # check if monomers
        deliboo = np.where( delj == 0)[0]
        
        dmti = scy.spatial.distance_matrix( ppmol[ None, dmmoli, : ], 
                                            pppol[ deliboo, :])    
        dmtidx = np.where( dmti <= clcutoff)

        for dmi in deliboo[ dmtidx[1] ]:
            if delj[ dmi] > 0:
                continue # monomer already discovered
            else:
                delj[ dmi] = idcl
                delij[ n1, dmi] = 1
                delij[ dmi, n1] = 1
                
# =============================================================================
#         for dmi in dmtidx[1]:
#             delj[ dmi] = idcl                
# =============================================================================
               
        
        ## go through molucules
        # deliboo = ~(deli > 0)
        deliboo = np.where( deli == 0)[0]
        
        dmti = scy.spatial.distance_matrix( ppmol[ None, dmmoli, : ], 
                                            ppmol[ deliboo, :] )    
        dmtidx = np.where( dmti <= clcutoff)

        for dmi in deliboo[ dmtidx[1] ]:
            if deli[ dmi] == 1:
                continue # molecule already used
            else :
                deli[ dmi] = 1 
                clmolfunc( dmi)
                
# =============================================================================
#         for dmi in dmtidx[1]:
#             deli[ dmi] = 1                 
#             clmolfunc( dmi)
# =============================================================================



    
    dfpolIdx = list(df.iloc[ np.isin( df.type, model.pol_type) ].index)
    dfmolIdx = list(df.iloc[ np.isin( df.type, list(set(model.alltypes) - set(model.pol_type))) ].index)
    pppol = df.iloc[dfpolIdx][['x','y','z']].values
    ppmol = df.iloc[dfmolIdx][['x','y','z']].values
    # ppall = df[['x','y','z']].values

    ranpolidx = random.sample( dfpolIdx, len(dfpolIdx))

    delj = np.zeros((pppol.shape[0]))
    delij = np.zeros((pppol.shape[0],pppol.shape[0]))
    idcl= 0
    for n1 in ranpolidx:
        if delj[n1] == 0:
            # print(n1)
            idcl+=1
            delj[n1] = idcl
            
            deli = np.zeros((ppmol.shape[0]))
            
            dmti = scy.spatial.distance_matrix( pppol[None,n1,:], 
                                                ppmol)    

            dmtidx = np.where( dmti <= clcutoff)

            for dmmoli in dmtidx[1]:
                deli[ dmmoli] = 1
                clmolfunc(dmmoli)

            




    dmtpol = scy.spatial.distance_matrix( pppol, 
                                        pppol ) 

    # pdb.set_trace()
    if mode == 'pcl':
        pcm = delij * np.exp( - dmtpol / d0)
        # pcm = delij * np.exp( - dmtpol / d0) * P1 + (1- delij) * np.exp( - dmtpol / d0) * P2

    elif mode == 'delpcl':
        pcm = delij * ( np.exp( + dmtpol / d0) -1)

    elif mode == 'delpcl2':
        alphd0 = (param.Nd0-1) / param.Nd0
        pcm = delij * (np.exp( + dmtpol / d0 * alphd0 ) -1)
        
    elif type( mode) is list:
        pcm = []
        for modei in mode:
            if modei == 'pcl':
                pcmi = delij * np.exp( - dmtpol / d0)
                # pcm = delij * np.exp( - dmtpol / d0) * P1 + (1- delij) * np.exp( - dmtpol / d0) * P2
        
            elif modei == 'delpcl':
                pcmi = delij * ( np.exp( + dmtpol / d0) -1)
        
            elif modei == 'delpcl2':
                alphd0 = (param.Nd0-1) / param.Nd0
                pcmi = delij * (np.exp( + dmtpol / d0 * alphd0 ) -1)            

            pcm += [pcmi]

    else:
        print( mode, 'is not a valid contact measure mode')
        raise

    return pcm


















def removeTails( chrstart, chrend, npol):
    
    # correct way
    # poltail = np.int_( np.round( nchrom ** (1/2)) )
    # wrong way - compatiblity
    if (type(npol) is list) and (len(npol) == 1):
        nchrom = (npol[0]+2) - 2* ( + npol[0] + 1)**(1/2.)
        poltail = np.int_( nchrom ** (1/2)) 
    elif (type(npol) is list) and (len(npol) > 1):
        nchrom = [(npoli+2) - 2* (  npoli + 1)**(1/2.) for npoli in npol]
        poltail = [ np.int_( nchromi ** (1/2)) for nchromi in nchrom]
    else:
        nchrom = (npol+2) - 2* ( + npol + 1)**(1/2.)
        poltail = np.int_( nchrom ** (1/2)) 


    return poltail








def contactM( ltypes, ssystem, methDic) :
    
    typdm = distM( ltypes, ssystem)
    
    typpc = np.int_( typdm < methDic['cutoff'] )
    
    return typpc, typdm
    











def readH5( strFile):
    
    # Check if all good    
    import h5py
    h5file = h5py.File( strFile, 'r')
    positions = h5file['particles/atoms/position/value']   
        
        





def readVTF():
    
    # =============================================================================
    #     import vtk
    #     reader = vtk.vtkUnstructuredGridReader(  'trajectory_'+str_sys+'.vtf')
    # =============================================================================

    pass



def writeCustom( ssystem):
    ppos = ssystem.part[:].pos
    typv = ssystem.part[:].type    
    
    df = pd.DataFrame(np.concatenate( (ppos, typv[:,None]), 1 ) )
    df['time'] = ssystem.time
    df['obs'] = 'pos'
    
    df.columns = ['x','y','z','type','time','obs']

    return df



def writeV( ssystem):
    pv = ssystem.part[:].v
    typv = ssystem.part[:].type    
    
    df = pd.DataFrame(np.concatenate( (pv, typv[:,None]), 1 ) )
    df['time'] = ssystem.time
    df['obs'] = 'v'
    
    df.columns = ['x','y','z','type','time','obs']

    return df



def readCustom( strfile):
    
    ssystem  = pd.read_csv( strFile, compression='gzip')
    
    return ssystem





def warmup( wmpDic):
        
    SIG = wmpDic['SIG']
    LJCUT = wmpDic['LJCUT']
    epsCapMax = wmpDic['epsCapMax']
    epsCapMin = wmpDic['epsCapMin']
    nnn = wmpDic['nnn']
    www = wmpDic['www']
    ttt = wmpDic['ttt']
    Ehf = wmpDic['Ehf']
    Ehi = wmpDic['Ehi']
    Enbf = wmpDic['Enbf']
    Enbi = wmpDic['Enbi']
    alltypes = wmpDic['alltypes']
    allpairs = wmpDic['allpairs']
    pairsb = wmpDic['pairsb']

    #
    Deps = epsCapMax - epsCapMin    
    lll = .1 # starting value
    mmm = .005 # starting value
    
    WARM_N_TIME = np.int_( (epsCapMax - epsCapMin) / epsCapMax / lll )
    WARM_STEPS = np.int_( epsCapMax * lll / mmm)
    
    # lll <= Deps / epsCapMax / nnn
    # WARM_STEPS >= www
    while lll > Deps / epsCapMax / nnn or WARM_STEPS < www or mmm > Deps / ttt:
        lll = lll / 1.5
        WARM_STEPS = np.int_( epsCapMax * lll / mmm)
        if lll > Deps / epsCapMax / nnn or WARM_STEPS < www or mmm > Deps / ttt:
            mmm = mmm / 2.
            WARM_STEPS = np.int_( epsCapMax * lll / mmm)
    
    #
    WARM_N_TIME = np.int_( (epsCapMax - epsCapMin) / epsCapMax / lll )
    ljhcap = np.arange(0,WARM_N_TIME,1) * mmm * WARM_STEPS + epsCapMin 
    #
    wmp_sam = np.int_( np.ones( WARM_N_TIME ) * WARM_STEPS )
    #
    wpT = WARM_N_TIME * WARM_STEPS
    
    
    # =============================================================================
    # Define warmup potentials
    # =============================================================================
    ##
# =============================================================================
#     ljh = {
#         'eps': np.zeros(( len(alltypes), len(alltypes), WARM_N_TIME)) ,
#         'sig': np.zeros(( len(alltypes), len(alltypes), WARM_N_TIME)) ,
#         'cut': np.zeros(( len(alltypes), len(alltypes), WARM_N_TIME)) ,
#         }
# =============================================================================
    ljh = {
        'eps': np.zeros(( alltypes[-1]+1, alltypes[-1]+1, WARM_N_TIME)) ,
        'sig': np.zeros(( alltypes[-1]+1, alltypes[-1]+1, WARM_N_TIME)) ,
        'cut': np.zeros(( alltypes[-1]+1, alltypes[-1]+1, WARM_N_TIME)) ,
        }

    # pdb.set_trace()
    
    for tyi in allpairs:
        ljh['eps'][ tyi[0], tyi[1], : ] = np.arange( Ehi[ tyi[0], tyi[1]] , Ehf[ tyi[0], tyi[1]], ( Ehf[ tyi[0], tyi[1]] - Ehi[ tyi[0], tyi[1]] ) / ( WARM_N_TIME ) ) [:WARM_N_TIME]
        ljh['sig'][ tyi[0], tyi[1], : ] = np.ones((WARM_N_TIME)) * SIG # .8
        ljh['cut'][ tyi[0], tyi[1], : ] = 2**(1/6.) * ljh['sig'][ tyi[0], tyi[1], : ]  
    
    ##
# =============================================================================
#     ljb = {
#         'eps': np.zeros(( len(alltypes), len(alltypes), WARM_N_TIME)) ,
#         'sig': np.zeros(( len(alltypes), len(alltypes), WARM_N_TIME)) ,
#         'cut': np.zeros(( len(alltypes), len(alltypes), WARM_N_TIME)) ,
#         }
# =============================================================================
    ljb = {
        'eps': np.zeros(( alltypes[-1]+1, alltypes[-1]+1, WARM_N_TIME)) ,
        'sig': np.zeros(( alltypes[-1]+1, alltypes[-1]+1, WARM_N_TIME)) ,
        'cut': np.zeros(( alltypes[-1]+1, alltypes[-1]+1, WARM_N_TIME)) ,
        }
    for tyi in pairsb:
        ljb['eps'][ tyi[0], tyi[1], : ] = np.arange( Enbi[ tyi[0], tyi[1]] , Enbf[ tyi[0], tyi[1]], ( Enbf[ tyi[0], tyi[1]] - Enbi[ tyi[0], tyi[1]]) / ( WARM_N_TIME)) [:WARM_N_TIME]
        ljb['sig'][ tyi[0], tyi[1], : ] = np.ones((WARM_N_TIME)) * SIG # np.arange( .8 , 1.5 * 2**(-1/6.), ( 1.5 * 2**(-1/6.) - .8) / ( WARM_N_TIME))  
        ljb['cut'][ tyi[0], tyi[1], : ] = np.arange( ljh['cut'][ tyi[0], tyi[1], -1 ] , LJCUT , ( LJCUT - ljh['cut'][ tyi[0], tyi[1], -1 ] ) / ( WARM_N_TIME))  [:WARM_N_TIME]
    
        
    return ljh, ljb, wmp_sam, ljhcap









def readConf( loadDict):

    try:
        print('reading config', loadDict['filename']+ '.gz' )
        df = pd.read_csv( loadDict['filename'] + '.gz', compression='gzip', index_col=0)
        # df = pd.read_csv( strStorage + '/umg/traj'+str(runi)+'_'+str_sys+'.gz', compression='gzip', index_col=0)
        # dftimesize = df.time.unique().size
        if loadDict['timeSample'][0] == -1:
            tme = np.sort( df.time.unique())[-1]
            df = df.set_index('time').loc[ tme ].reset_index()
            
        elif type( loadDict['timeSample'] ) is str:
            df = df.set_index('simulation stage').loc[ loadDict['timeSample'] ].reset_index()
                
            tme = np.sort( df.time.unique())[-1]
            df = df.set_index('time').loc[ tme ].reset_index()            
            
        else:
            df = df.set_index('time').loc[ loadDict['timeSample']].reset_index()
        
        
        df['index'] = df.index
        df.type = df.type.apply( int)
        
    except:
        print('config not found. Skipping...')
        raise
        
 
    
    
    return df # , dftimesize



def readConf2( loadDict):

    try:
        print('reading config', loadDict['filename']+ '.gz' )
        df = pd.read_csv( loadDict['filename'] + '.gz', compression='gzip', index_col=0)
        # df = pd.read_csv( strStorage + '/umg/traj'+str(runi)+'_'+str_sys+'.gz', compression='gzip', index_col=0)
        # dftimesize = df.time.unique().size
        dftimes = df.time.unique()

        if loadDict['timeSample'][0] == -1:
            tme = np.sort( df.time.unique())[-1]
            df = df.set_index('time').loc[ tme ].reset_index()
            
        elif type( loadDict['timeSample'] ) is str:
            df = df.set_index('simulation stage').loc[ loadDict['timeSample'] ].reset_index()
                
            tme = np.sort( df.time.unique())[-1]
            df = df.set_index('time').loc[ tme ].reset_index()            
            
        else:
            df = df.set_index('time').loc[ loadDict['timeSample']].reset_index()
        
        
        df['index'] = df.index
        df.type = df.type.apply( int)
        
    except:
        print('config not found. Skipping...')
        raise
    
    
    return df, dftimes








def logSamp( samDict):
    
    ## logarithmic extension
    mint = samDict['mint']
    maxt = samDict['maxt']
    tit = samDict['tit']

    if ('maxt2' in samDict.keys()) and (samDict['maxt2'] is not False):
        maxt2 = samDict['maxt2']
    
        Dn = np.ceil( np.log( (maxt2 - mint) / mint) / np.log( maxt / mint ) * tit) - tit

        # tp = np.int_( mint * (maxt/mint) ** ( np.arange(tit, tit +Dn +1)/tit) ) + mint
        tp = np.int_( mint * (maxt/mint) ** ( np.arange(0, tit +Dn +1)/tit) ) + mint

    else:
        tp = np.int_( mint * (maxt/mint) ** ( np.arange(0, tit +1)/tit) ) + mint
        
    
    return tp    
    
    













def lopeAnalys( lopePairs, system, anistype):
    ### lope analysis
    lpdf = pd.DataFrame( lopePairs)
    lpdf.sort_values(by=1, inplace=True)
    lpdf.columns = ['poltype', 'anistype', 'strand']
    
    dimask = np.isin( system.part[:].type  , anistype)
    sysanis = pd.DataFrame({
        'id' : system.part[:].id[ dimask ],
        'type' : system.part[:].type[ dimask ]
            })
    
    sysanis['dimPairs'] = np.int_( np.arange(1.1, sysanis.shape[0]/2. + 1.1,.5) )
    
    lloop = lpdf.merge( sysanis, how= 'left', left_on = 'anistype', right_on = 'id' )
    
    llooptmp = lloop.groupby('dimPairs').agg( len).reset_index()[['dimPairs','id']]
    llooptmp = llooptmp[ llooptmp ['id'] == 2 ]
    # llooptmp.columns = ['dimPairs','id']
    lloop2 = lloop.merge( llooptmp['dimPairs'], how='right', on='dimPairs' )
    
    lloop3 = lloop2.pivot_table( columns = 'type', index='dimPairs', values='poltype')
    lloop3['loopDim'] = lloop3[ anistype[0]] - lloop3[ anistype[1]]
    
    lloop3['loopDim'].mean(), lloop3['loopDim'].std()
    

    return lloop3













# =============================================================================
# Save configuration
# =============================================================================


def confBegin( ssystem, model, espiowriter):
    if model.str_write=='vmd':
        fp = open( model.filenameTraj +'.vtf', mode='w+t')

        # write structure block as header
        espiowriter.vtf.writevsf( ssystem, fp)
        
        dft = 0
             

    elif model.str_write == 'h5md':
        os.system('rm ' + model.filenameTraj +'.h5')
        fp = espiowriter.h5md.H5md(filename= model.filenameTraj + '.h5'
                       , write_pos=True
                       , write_vel=True
                       , write_species = True
                       )
        fp.write()
        dft = 0
        
    elif model.str_write == 'custom':
        fp = open( model.filenameTraj +'.vtf', mode='w+t')

        # write structure block as header
        espiowriter.vtf.writevsf( ssystem, fp)

        # save csv
        dft = writeCustom( ssystem)




    return fp, dft





def confSave( ssystem, model, espiowriter, fp, dft, strVel = False, precision=None):
    for wri in range(7):
        try:
            if model.str_write=='vmd':
                espiowriter.vtf.writevcf( ssystem, fp)
                
            elif model.str_write == 'h5md':
                espiowriter.h5.write()            
                
            elif model.str_write == 'custom':
                presTime = time.time()
                if (presTime - model.startTime > 5 * 60 * 60) and \
                    ( (model.str_tsam == 'log') or \
                     (( 'strFlagSaveInterm' in dir(model)) and (model.strFlagSaveInterm is True))) : 
                    try: 
                        espiowriter.vtf.writevcf( ssystem, fp)
                        fp.close()
                    except:
                        fp = open( model.filenameTraj +'.vtf', mode='a')
                        espiowriter.vtf.writevcf( ssystem, fp)
                        fp.close()

                    # save csv
                    dftmp = writeCustom( ssystem)                
                    dft = dft.append( dftmp )     
                    
                    # save velocities?
                    if strVel :
                        dfv = writeV( ssystem)
                        dft = dft.append( dfv )   
                    else:
                        print(' -> Saving configuration without velocities')
                        
                    # save only at a lower precision?
                    if precision is not None:
                        dft[['x','y','z']] = np.around( dft[['x','y','z']].values, precision )
                        
                        
                    dft.to_csv( model.filenameTraj +'.gz', compression='gzip')      

                else:
                    espiowriter.vtf.writevcf( ssystem, fp)
                    fp.flush()
                    # append csv
                    dftmp = writeCustom( ssystem)                
                    dft = dft.append( dftmp )                     
                    
                    # save velocities?
                    if strVel :
                        dfv = writeV( ssystem)
                        dft = dft.append( dfv )     
                        
                    # save only at a lower precision?
                    if precision is not None:
                        dft[['x','y','z']] = np.around( dft[['x','y','z']].values, precision )
                        
                                                


            elif model.str_write == 'pdb':
                # espiowriter = universe
                u = espiowriter
                u.atoms.write("system.pdb")

                
            elif model.str_write == 'gromacs':
                eos, u, W = espiowriter
                u.load_new(eos.trajectory)  # load the frame to the MDA universe
                W.write_next_timestep(u.trajectory.ts)  # append it to the trajectory
                                            
                    
            break
        
        except:
            wrimin = random.randrange(10,30)
            time.sleep( 60 * wrimin)
            print('Writing attempt n.',wri,'failed. Retrying in ',wrimin,'minutes...')
     
        if wri == 7:
            print('All writing attempts failed. Save what you can and get out of here.')
            raise
        

    return fp, dft




def confEnd( ssystem, model, espiowriter, fp, dft, precision=None ):
    for wri in range(7):
        try:
            if model.str_write=='vmd':
                espiowriter.vtf.writevcf( ssystem, fp)
                fp.close()
                
            elif model.str_write == 'h5md':
                espiowriter.h5.write() 
                espiowriter.h5.close()            
                
            elif model.str_write == 'custom':
                print('Saving configuration with velocities')
                try: 
                    espiowriter.vtf.writevcf( ssystem, fp)
                    fp.close()
                except:
                    fp = open( model.filenameTraj +'.vtf', mode='a')
                    espiowriter.vtf.writevcf( ssystem, fp)
                    fp.close()

                # save csv
                dftmp = writeCustom( ssystem)                
                dfv = writeV( ssystem)
        
                dft = dft.append( dftmp )                     
                dft = dft.append( dfv )                     

                # save only at a lower precision?
                if precision is not None:
                    dft[['x','y','z']] = np.around( dft[['x','y','z']].values, precision )
                        
                dft.to_csv( model.filenameTraj +'.gz', compression='gzip')      


            elif model.str_write == 'pdb':
                # espiowriter = universe
                u = espiowriter
                u.atoms.write("system.pdb")

                
            elif model.str_write == 'gromacs':
                eos, u, W = espiowriter
                u.load_new(eos.trajectory)  # load the frame to the MDA universe
                W.write_next_timestep(u.trajectory.ts)  # append it to the trajectory
                    
    
            break
        
        except:
            wrimin = random.randrange(10,30)
            time.sleep( 60 * wrimin)
            print('Writing attempt n.',wri,'failed. Retrying in ',wrimin,'minutes...')
     
        if wri == 7:
            print('Final config I/O failed. Final configuration lost')    





def confEndWarmupError( model, espiowriter, fp ):
    if model.str_write=='vmd':
        fp.close()
        
    elif model.str_write == 'h5md':
        espiowriter.h5.close()            
        os.system('rm ' + strStorage + '/traj'+ procid +'_'+str_sys+'.h5')         
        
    elif model.str_write == 'custom':
        fp.close()     




def saveVTK( ssystem, fn):
    # write to VTK
    ssystem.part.writevtk("part_type_0_1.vtk", types=[0, 1])
    ssystem.part.writevtk("part_type_2.vtk", types=[2])
    ssystem.part.writevtk("part_all.vtk")    





###========================================================================
# Measure
###========================================================================  

def obsSave( ssystem, model, dynName, exept=False):
    
    if exept :
        obss = pd.DataFrame(
            data = {
                model.obs_cols[0]: [ssystem.time] ,
                model.obs_cols[1]: [  np.nan ] ,
                model.obs_cols[2]: [  np.nan ] ,
                }
            )
    
    else:
        energies = ssystem.analysis.energy()

        obss = pd.DataFrame(
            data = {
                model.obs_cols[0]: [ssystem.time] ,
                model.obs_cols[1]: [energies['kinetic'] / (1.5 * np.sum( model.Npart ))] ,
                model.obs_cols[2]: [energies['total']] ,
                }
            )

    for idtyp, tyi in enumerate(model.typev):
        try:
            rg2 = ssystem.analysis.gyration_tensor( p_type= tyi)['Rg^2']    
            
            if rg2 == 0: obss[ model.obs_cols[idtyp + 3]] = np.nan
            else: obss[ model.obs_cols[idtyp + 3]] = rg2

        except:
            obss[ model.obs_cols[idtyp + 3]] = np.nan
            
            

    obss['simulation stage'] = dynName

    return obss







def obsBegin( ssystem, model):
        
    return obsSave( ssystem, model, 'start')

    

def obsInterm( ssystem, model, obss_all, dynName, exept = False):
    
    obss_all = obss_all.append( obsSave( ssystem, model, dynName, exept ) )

    return obss_all




def obsEnd(obss_all, model, env):
    # pickle
    if model.str_saveMode == 'pickle':
        with open(env.strStorEspr + '/' + model.str_syst + '/obs/pickle_' + model.str_syst, 'wb') as handle:
            pickle.dump( [rg2, typev, stime]
                        , handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # pandas table        
    elif model.str_saveMode == 'pandas':
        try:
            data_df_old = pd.read_csv( env.strStorEspr + '/' + model.str_syst + '/obs/obs_' + model.str_syst +'_'+model.str_param +'.csv', index_col=None)
            data_df_old.append( obss_all).to_csv( env.strStorEspr + '/' + model.str_syst + '/obs/obs_' + model.str_syst +'_'+model.str_param + '.csv', index=False)
            print('Warning: file appended to an existing one.')
        except:
            obss_all.to_csv( env.strStorEspr + '/' + model.str_syst + '/obs/obs_' + model.str_syst +'_'+model.str_param + '.csv', index=False)      

    # pandas table        
    elif model.str_saveMode == 'pandas+w':
        obss_all.to_csv( env.strStorEspr + '/' + model.str_syst + '/obs/obs_' + model.str_syst +'_'+model.str_param + '.csv', index=False)      






## new obs measure
def obsLopeSave(obss_all, model, env):
    
    # conf ID
    obss_all['conf'] = model.procid
    obss_all['param']= model.str_param
    

    
    # pickle
    if model.str_saveMode == 'pickle':
        with open(env.strHome + '/pickle_' + model.str_syst, 'wb') as handle:
            pickle.dump( [rg2, typev, stime]
                        , handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # pandas table        
    elif model.str_saveMode == 'pandas':
        try:
            data_df_old = pd.read_csv( env.strHome + '/obsLope_' + model.str_syst +'.csv', index_col=None)
            data_df_old.append( obss_all).to_csv( env.strHome + '/obsLope_' + model.str_syst + '.csv', index=False)
            print('Warning: file appended to an existing one.')
        except:
            obss_all.to_csv( env.strHome + '/obsLope_' + model.str_syst + '.csv', index=False)      



def obsLopeSave2(obss_all, model, env, strKind, When='interm'):
    
    # conf ID
    # obss_all['conf'] = model.procid
    # obss_all['param']= model.str_param

    presTime = time.time()
    if (When == 'end') or ( (presTime - model.startTime > 5 * 60 * 60) and \
                    ( (model.str_tsam == 'log') or \
                     (( 'strFlagSaveInterm' in dir(model)) and (model.strFlagSaveInterm is True))) ): 
        print(' -> Saving '+strKind+' obs')

    
        # pickle
        if model.str_saveMode == 'pickle':
            with open(env.strStorEspr + '/' + model.str_syst + '/obs/pickle_' + model.str_syst, 'wb') as handle:
                pickle.dump( [rg2, typev, stime]
                            , handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        # pandas table        
        elif model.str_saveMode == 'pandas':
            try:
                data_df_old = pd.read_csv( env.strStorEspr + '/' + model.str_syst + '/obs/obs'+strKind+'_' + model.str_syst +'_'+model.str_param+'_c' + model.procid +'.csv.gz', index_col=None, compression='gzip')
                data_df_old.append( obss_all).to_csv( env.strStorEspr + '/' + model.str_syst + '/obs/obs'+strKind+'_' + model.str_syst +'_'+model.str_param+ '_c' + model.procid +'.csv.gz', index=False, compression='gzip')
                print('Warning: file appended to an existing one.')
            except:
                obss_all.to_csv( env.strStorEspr + '/' + model.str_syst + '/obs/obs'+strKind+'_' + model.str_syst +'_'+model.str_param+ '_c' + model.procid +'.csv.gz', index=False, compression='gzip')      

        # pandas table        
        elif (model.str_saveMode == 'pandas+w') and (When=='start'):
            obss_all.to_csv( env.strStorEspr + '/' + model.str_syst + '/obs/obs'+strKind+'_' + model.str_syst +'_'+model.str_param+ '_c' + model.procid +'.csv.gz', index=False, compression='gzip')      



        obss_all = pd.DataFrame([])    
    else:
        pass
        
    return obss_all












# =============================================================================
# conf I/O
# =============================================================================










#####

def bin2bead( df ):
    tmpl = list( df.type.astype(int).values )
    tmpl.sort()
    return tmpl


def bin2type( df, binning, typv):
    df['Index'] = df.index
    df = df.set_index('type').loc[ typv]
    df = df.sort_values( 'Index').reset_index()    
    df['Index'] = df.index
    # 
    binning['Index'] = binning.reset_index().index
    
    #
    ppos02 = df.merge( binning, on='Index', how='inner')
    beatypl = ppos02[['type','part']].groupby('part').apply( bin2bead).tolist()
    beatypl.sort()
    return list( beatypl  for beatypl,_ in itertools.groupby(beatypl ))



def bin2type2( binning):
    partBeads = binning[['type','part','polyid']].groupby(['polyid','part'], sort=False).apply( bin2bead)
    
    beatypl = partBeads.tolist()
    beatypl.sort()
    
    return list( beatypl  for beatypl,_ in itertools.groupby(beatypl )), partBeads



def getUniqLL( LL):
    return list( beatypl  for beatypl,_ in itertools.groupby( LL ))



def getUniqLL2( LL):
    return [list(x) for x in set(tuple(x) for x in LL)]

















def makePPfromHist( dmhist, dcutoff = 3 ):
    normval = dmhist[0,1,1,:].sum()
    
    
    prob = np.zeros((dmhist.shape[1],dmhist.shape[2]))
    for di1 in range( dmhist.shape[1]):
        print(di1,'/',dmhist.shape[1])
        for di2 in range( di1, dmhist.shape[2]):
           
            yhat = scygnal.savgol_filter( dmhist[ 0, di1, di2, :]
                                            , 5, 3) # window size 51, polynomial order 3
            yhat[yhat < 0] = 0

            prob[ di1, di2] = yhat[:dcutoff].sum() / normval



    prob = prob.T + np.triu( prob, 1 )



    if False:
        f2 = scy.interpolate.interp1d( range(11), dmhist[ 0, di1, di2, :], kind='cubic')
        f2( range(11))
        f2( range(3) ).sum()/10


    if False:
        def gaussLS( par, x ):
            return par[2] * np.exp( - ((x- par[0])/par[1])**2)
        
        
        def resiGaussLS( par, x, y ):
            return par[2] * np.exp( - ((x- par[0])/par[1])**2) - y
                
        nlsq = scy.optimize.least_squares( resiGaussLS, 
                                          np.array([ dmhist[ 0, di1, di2, :].argmax() , 1,1])
                                         , bounds = np.array([[0,0,.5],[10,10,10]])
                                         , args=(range(11), dmhist[ 0, di1, di2, :]/10 )
                                         )
        nlsq.x
        nlsq.cost
        
        chi, ppv = scy.stats.chisquare( dmhist[ 0, di1, di2, :], gaussLS( nlsq.x, range(11) ) )
        if ppv > .9: 
            print('prendi prob from stats.cdf')
        else :
            print('prendi prob from trivial cutoff')         

    if False:      
        scy.stats.chisquare( np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 7], dtype=np.float32), 
                        np.array([.0001, .0001, .0001, .0001, .0001, .0001, .0001, .0001, .0001, 2, 7], dtype=np.float32) )
    
        # @vectorize
        def pdf_int_fit( tau, f, thet):
            return np.exp(-N * tau) / ( N * self.T + A1/f * ( np.sin( 2 * np.pi * self.T + thet) - np.sin(thet) ) ) * quadrature( pdf_integrand_fit, 0, self.T, args=(tau,f,thet))[0]
        vPdf_int_fit = np.vectorize(pdf_int_fit)
    
        def resid(x):
            return vPdf_int_fit( self.tau_rho, x[0],x[1]) - self.tau_pdf_beat[0]
    
        nlsq = sc.optimize.least_squares( resid, np.array([fi,thi]), bounds = np.array([[fv[0],thv[0]],[fv[-1],thv[-1]]]) )
        # costt = nlsq.cost
        if nlsq.cost < costf:
            costf = nlsq.cost
            fb, thb = fi, thi
            nlsqf = nlsq    
        
        
        
        
        loc, scal = scy.stats.norm.fit( cDval[ 0, 1, 5, :] )
        pdfobs = scy.stats.norm.pdf( list(range(10)), loc=loc, scale=scal)
        
        
        
        dmi = np.random.uniform(size=(2,3,2,5))
        
        dcond = dmi[0,:,:,0] < .3
        
        dmi[ 0,dcond, 0] = dmi[ 0,dcond, 0] +1
        
        
        dmii = scy.stats.norm.rvs( 0,1,1000 )
        parr = scy.stats.norm.fit(dmii)
        # scy.stats.rv_continuous.fit( dmi) ?
        pdfobs = scy.stats.norm.pdf( [.1,.4,.7], loc=parr[0], scale=parr[1])
        scy.stats.norm.cdf([ .5], loc=parr[0], scale=parr[1])
        C = 1 / np.sum(pdfobs * .3)
        pdfobs * C
        
        dminins = np.histogram( dmii, bins=[.0,.2,.5,.8], density=True)
        chi, ppv = scy.stats.chisquare( dminins[0], pdfobs * C )
        if ppv > .9: 
            print('prendi prob from stats.cdf')
        else :
            print('prendi prob from trivial cutoff')
        
        ##  this does not interpolate over min/max values from histogram
        histat = scy.stats.rv_histogram( (dmii[:5]+2, multil[:5+1,0]))
        histat.cdf(10)
    

    return prob









def confChooRan( runidv, nsampl, rseed, strTimel, tpveq, ranlist=None):
    
    pdb.set_trace()
        
    # check time condition
    if 'condition' in re.split('\+', strTimel[0]):
        timev = df.time.unique()
        tpv = np.argwhere( np.isclose( timev, te, atol=tatol ))
        if 'last' in re.split('\+', strTimel[0]):
            tpv = tpv[-1]

    else:
        tpv = tpveq        
    

    
    if (nsampl > 0 ) & ( ranlist is None):
        # select the platea
        fullplat = list( itertools.product( runidv, tpv) )
        
        random.seed( rseed)
        runidvsampl = random.sample( fullplat, nsampl)
    
        ranconftimel = pd.DataFrame( runidvsampl).groupby(0)[1].apply(list).reset_index().values.tolist()        

    elif (nsampl == 1 ) & ( type(ranlist) is list):
        # select the platea
        ranconftimel, ranelem = ranlist
        subsunidv = list( set( runidv ) - set( [ cossi[0] for cossi in ranconftimel] ))
        fullplat = list( itertools.product( subsunidv, tpv) )

        runidvsampl = random.sample( fullplat, nsampl)
        runidvsampl = pd.DataFrame( runidvsampl).groupby(0)[1].apply(list).reset_index().values.tolist()        

        ranconftimel[ ranconftimel.index( ranelem)] = runidvsampl[0]
                
    else:
        # select the platea
        fullplat = list( itertools.product( runidv, tpv) )
        ranconftimel = pd.DataFrame( fullplat).groupby(0)[1].apply(list).reset_index().values.tolist()        

        
            
    return ranconftimel





def selectSimulTime( strTimel, tpveq):
    # check time condition
    if 'condition' in re.split('\+', strTimel[0]):
        timev = df.time.unique()
        tpv = np.argwhere( np.isclose( timev, te, atol=tatol ))
        if 'last' in re.split('\+', strTimel[0]):
            tpv = tpv[-1]

    else:
        tpv = tpveq 
        
    return tpv




def confChooRan2( confdall, strDataset, nsampl, rseed, strTimel, ranlist=None):
    
    # pdb.set_trace()
    if (nsampl > 0 ) & ( ranlist is None):
        fullplat = []
        # select the platea
        for idsds, sds in enumerate(strDataset):
            runidv = confdall[ sds ][0]
            runidstr = confdall[ sds ][1]
            tpveq = confdall[ sds ][2] # [-1,-6]    
        
            # check time condition
            tpv = selectSimulTime( strTimel, tpveq)
        
            #        
            fullplattmp = list( itertools.product( [idsds], runidv, tpv) )
        
            fullplat += fullplattmp

    
        ###    
        random.seed( rseed)
        runidvsampl = random.sample( fullplat, nsampl)
    
        ranconftimel = pd.DataFrame( runidvsampl).groupby([0,1])[2].apply(list).reset_index().values.tolist()
        
        # ranconftimel[0] = [0, 426, [-1, -4, -7, -10]]
    
    
    elif (nsampl == 1 ) & ( type(ranlist) is list) :
        ranconftimel, ranelem = ranlist

        runidv = confdall[ strDataset[1] ][0]
        runidstr = confdall[ strDataset[1] ][1]
        tpveq = confdall[ strDataset[1] ][2] # [-1,-6]   
        
        # check time condition
        tpv = selectSimulTime( strTimel, tpveq)
        
    
        # select the platea
        subsunidv = list( set( runidv ) - set( [ cossi[1] for cossi in ranconftimel] ))
        if len(subsunidv) == 0:
            ranconftimel[ ranconftimel.index( ranelem)] = [0,0,[]]

        else:
            fullplat = list( itertools.product( [strDataset[0] ], subsunidv ) )
            # randomly sub sampl it
            # runidvsampl = random.sample( fullplat, nsampl) # old working
            runidvsampl = list(random.sample( fullplat, 1 )[0])
    
            runidvsampl = runidvsampl + [random.sample( tpv, len(ranelem[2]) )]
            ranconftimel[ ranconftimel.index( ranelem)] = runidvsampl
    
    else:
        # select the full platea
        print('Full config\'s platea selected')
        fullplat = []
        for idsds, sds in enumerate(strDataset):
            runidv = confdall[ sds ][0]
            runidstr = confdall[ sds ][1]
            tpveq = confdall[ sds ][2] # [-1,-6]    
        
            # check time condition
            tpv = selectSimulTime( strTimel, tpveq)
        
            #        
            fullplattmp = list( itertools.product( [idsds], runidv, tpv) )
        
            fullplat += fullplattmp            
    
    
        ranconftimel = pd.DataFrame( fullplat).groupby([0,1])[2].apply(list).reset_index().values.tolist()        
    
    
    return ranconftimel






def makeHist( strDsd, str2model, nsampl, baseHistBin=1., settDictInfoFN='settDict', rseed=42, strTimel = ['eq'], strMeth = 'allTypes', dbinmax=10 ):

    #
    exec( 'from ' + settDictInfoFN + ' import confdall', globals() )
    # 
    histD = {}
    # guarantee reproducibility
    np.random.seed(rseed)
    
    #
    for sdsk, strDataset in strDsd.items():
        print( 'updating', sdsk, 'with', strDataset) 
        exec( 'import ' + str2model[ sdsk ]  + ' as model', globals() )
        
        ##
        dbin0 = model.SIG * baseHistBin
        dlis = list(zip(range(0,dbinmax,1), range(1,dbinmax +1,1) )) + [[dbinmax,np.Inf]]
        multil = np.array(dlis) * dbin0        
        
        ##
        nid = 0 # len( runidv) * len( tpv)
    
        ranconftimel = confChooRan2( confdall, strDataset, nsampl, rseed, strTimel)

        # loop over configs
        for idrt, ranli in enumerate( ranconftimel ):
        
            idsds = ranli[0]
            sds = strDataset[idsds]
            
            runidv = confdall[ sds ][0]
            runidstr = confdall[ sds ][1]
            tpveq = confdall[ sds ][2] # [-1,-6]
            
            if len( confdall[ sds ]) > 3 and type(confdall[ sds ][3]) is str:
                strModel = confdall[ sds ][3]
            else:
                strModel = 'bloom2021'  # brackley2016, brackley2016stiff, bloom2021
    
    
            # check if config exists
            boolConf = True
            while boolConf:
                try:
                    print('read conf ' +str(idrt+1) + '/' + str(len(ranconftimel)) + ': ' + runidstr + str( ranli[1]) + ' t'+str(ranli[2]))
                    
                    if env.strCluster == 'UbuntuHome' : 
                        confFN = env.strStorage + '/umg/traj'+ runidstr + str( ranli[1] )+'_'+strModel+'.gz'
                    elif env.strCluster == 'GDWbarbieri' :
                        confFN = env.strStorage + '/traj'+ runidstr + str( ranli[1] )+'_'+strModel+'.gz'

                    df = pd.read_csv( confFN, compression='gzip', index_col=False)
                        
                    boolConf = False
                except:
                    # pdb.set_trace()
                    print('file ' + confFN +' not found')
                   
                    # ranconftimel = confChooRan( runidv, 1, rseed, strTimel, tpveq, [ranconftimel,ranconftimel[idrt] ])
                    ranconftimel = confChooRan2( confdall, [idsds,sds] , 1, rseed, strTimel, [ranconftimel, ranli ])
                    ranli = ranconftimel[ idrt]
 
                    if len(ranli[2]) == 0: 
                        print('-> skipping to next conf')
                        break
                    else: 
                        print('-> found another conf')
                        continue
                    
            
                # loop over times
                for idtv, tmtyp in enumerate( strTimel):
                    for idt, tmei in enumerate( ranli[2]):
    
                        tmeall = df.time.unique()
                        tmev = tmeall[ tmei ]
    
                        dft = df.set_index('time').loc[ tmev].reset_index()
                        
                        # check if there's velocities
                        # if ('obs' in dft.columns) and ( dft[ dft[ 'obs'] == 'v'].shape[0] > 0 ):
                        if ('obs' in dft.columns) and np.isin( 'v', dft[ 'obs'].unique()):
                            dft = dft[ dft[ 'obs'] == 'pos']
                        
                        # check if there's duplicates
                        if len( confdall[ sds ]) > 3 and type(confdall[ sds ][3]) is int:
                            dft = dft.drop_duplicates(['x','y','z'])
                            

                        # dmm2 = distM3_1( dft, model.pol_type, model.binning, model.chrstart, model.chrend, model.npol)
                        dmm2 = distM3_1p1( dft, model.pol_type, model.binning, model.chrstart, model.chrend, model.npol)
                        # pdb.set_trace()
                
                
                        if idtv == 0 and nid == 0:
                            dmhist = np.zeros( ( len( strTimel), dmm2.shape[0], dmm2.shape[1], len(multil)) , dtype=np.int32)
                           

                        
                        ### accumulate counts over previous configs   
                        for idxc, cutmulti in enumerate(multil):
                            dcond = ( dmm2 >= cutmulti[0] * dbin0 ) & ( dmm2 < cutmulti[1] * dbin0)
                            dmhist[ idtv, dcond, idxc] = dmhist[ idtv, dcond, idxc] +1                            
                        

                
            
            
            nid = nid + len( ranli[2] ) 
            
        #
        histD.update( { sdsk+'_n'+str(nid)+ '_hist_bin' + str(round(baseHistBin,2)) : dmhist} )
    
    return histD










def makeCounts4( strDsd, str2model, nsampl, bondLengthFraction=2., settDictInfoFN='settDict', rseed=42, strTimel = ['eq'], strMeth = 'allTypes', distMode='3.1' ):
    # strTimel = ['eq'] # ['condition+last','eq']
    #
    exec( 'from ' + settDictInfoFN + ' import confdall', globals() )

    # 
    countD = {}
    dmD = {}
    
    #
    dmunit = 1
    
    # guarantee reproducibility
    np.random.seed(rseed)
    
    #
    for sdsk, strDataset in strDsd.items():
        print( 'updating', sdsk, 'with', strDataset) 
        exec( 'import ' + str2model[ sdsk ]  + ' as model', globals() )
        try:
            exec( 'model = model.'+ re.sub('_model','',str2model[ sdsk ]) +'()', globals())
        except:
            pass

        dcutoff2 = model.LJCUT * bondLengthFraction # model.bond_length * bondLengthFraction
        d0 = round( - 2 * model.SIG / np.log(.5), 3) # model.SIG * 3 # 3 # dcutoff2 / 2
        
        nid = 0 # len( runidv) * len( tpv)
    
        ranconftimel = confChooRan2( confdall, strDataset, nsampl, rseed, strTimel)

        # loop over configs
        for idrt, ranli in enumerate( ranconftimel ):
        
            idsds = ranli[0]
            sds = strDataset[idsds]
            
            runidv = confdall[ sds ][0]
            runidstr = confdall[ sds ][1]
            tpveq = confdall[ sds ][2] # [-1,-6]
            
            if len( confdall[ sds ]) > 3 and type(confdall[ sds ][3]) is str:
                strModel = confdall[ sds ][3]
            else:
                strModel = 'bloom2021'  # brackley2016, brackley2016stiff, bloom2021
    
            if type( runidstr) is list:
                readPattern = env.strStorEspr + '/'+ runidstr[0] + '/traj_' + runidstr[1] + '_%d_' + runidstr[2] + '_' +runidstr[3] + '.gz'
                printPattern = 'read conf '+str(idrt+1) + '/%d: '+ runidstr[0] + '/traj_' + runidstr[1] + '_%d_' + runidstr[2] + '_' +runidstr[3] + ' at t=%s'
            else:
                if env.strCluster == 'UbuntuHome' : 
                    readPattern = env.strStorage + '/umg/traj'+ runidstr + '%d_' + strModel+'.gz'
                elif env.strCluster == 'GDWbarbieri' :
                    readPattern = env.strStorage + '/traj'+ runidstr + '%d_' +strModel+'.gz'     
    
                printPattern = 'read conf '+str(idrt+1) + '/%d: '+ runidstr + '%d_' + strModel + ' at t=%s'
    
            # check if config exists
            boolConf = True
            while boolConf:
                try:
                    
                    if False:
                        print('read conf ' +str(idrt+1) + '/' + str(len(ranconftimel)) + ': ' + runidstr + str( ranli[1]) + ' t'+str(ranli[2]))
                        if env.strCluster == 'UbuntuHome' : 
                            confFN = env.strStorage + '/umg/traj'+ runidstr + str( ranli[1] )+'_'+strModel+'.gz'
                        elif env.strCluster == 'GDWbarbieri' :
                            confFN = env.strStorage + '/traj'+ runidstr + str( ranli[1] )+'_'+strModel+'.gz'
                    else:
                        confFN = readPattern % ( ranli[1] )
                        print( printPattern % ( len(ranconftimel), ranli[1] , str(ranli[2]) ) )


                    ##
                    df = pd.read_csv( confFN, compression='gzip', index_col=False)
                        
                    boolConf = False
                except:
                    # pdb.set_trace()
                    print('file ' + confFN +' not found')
                   
                    # ranconftimel = confChooRan( runidv, 1, rseed, strTimel, tpveq, [ranconftimel,ranconftimel[idrt] ])
                    ranconftimel = confChooRan2( confdall, [idsds,sds] , 1, rseed, strTimel, [ranconftimel, ranli ])
                    ranli = ranconftimel[ idrt]
 
                    if len(ranli[2]) == 0: 
                        print('-> skipping to next conf')
                        break
                    else: 
                        print('-> found another conf')
                        continue
                    
            
                # loop over times
                for idtv, tmtyp in enumerate( strTimel):
                    for idt, tmei in enumerate( ranli[2]):
    
                        tmeall = df.time.unique()
                        tmev = tmeall[ tmei ]
    
                        dft = df.set_index('time').loc[ tmev].reset_index()
                        
                        # check if there's velocities
                        # if ('obs' in dft.columns) and ( dft[ dft[ 'obs'] == 'v'].shape[0] > 0 ):
                        if ('obs' in dft.columns) and np.isin( 'v', dft[ 'obs'].unique()):
                            dft = dft[ dft[ 'obs'] == 'pos']
                        
                        # check if there's duplicates
                        if len( confdall[ sds ]) > 3 and type(confdall[ sds ][3]) is int:
                            dft = dft.drop_duplicates(['x','y','z'])
                
                            
                
                        if distMode == '3.4':
                            dmm2, dftype = distM3_4( dft, model.pol_type)
                        if distMode == '3.2':
                            dmm2, dftype = distM3_2( dft, model.pol_type, model.binning, model.chrstart, model.chrend, model.npol)
                        elif distMode == '3.1.1':
                            dmm2 = distM3_1p1( dft, model.pol_type, model.binning, model.chrstart, model.chrend, model.npol)
                        elif distMode == '3.1':
                            dmm2 = distM3_1( dft, model.pol_type, model.binning, model.chrstart, model.chrend, model.npol)
                        else:
                            dmm2 = distM3( dft, model.pol_type)
                            
                
                
                        ###
                        if idtv == 0 and nid == 0 and idt == 0:
                            dmm = np.zeros( ( len( strTimel), dmm2.shape[0], dmm2.shape[1]) )
                            count = np.zeros( ( len( strTimel), dmm2.shape[0], dmm2.shape[1]) )
                            
                            
                            
                        ### accumulate distances over previous configs   
                        dmm[ idtv, :, :] = dmm[ idtv, :, :] +  dmm2                    
                        
                        
                        ###### hic counts noise part ####### 
                        typl = dft.type.values
    
                        # take only upper part
                        dmm2[ np.tril_indices( dmm2.shape[0], k=-1) ] = np.inf
                        
                        # crosslink only beads that have a real chemical bond    
                        dmcfidx = (dmm2 > dcutoff2)
                        dmnnzidx = (~dmcfidx).nonzero()
    
                        #
                        nnzidx = list(zip(dmnnzidx[0],dmnnzidx[1]))
                    
                        # sort contacts according to:
                            # random drawing
                            # distance value
                            # distance value (stochastically)
                        ranunil = np.random.uniform(0,1, size= len(nnzidx))
                        dmr = np.exp( - dmm2[dmnnzidx] / d0) / ranunil
                    
                        if strMeth == 'onlySameType':
                            dml2 = [ [dmi[0],dmi[1],dmm2[nnzidx[idx]], dmr[idx]] for idx, dmi in enumerate(nnzidx) if (typl[dmi[1]]==typl[dmi[2]]) ]
                            idxr = sorted( dml2, key=lambda item: item[3], reverse=True)
                        elif strMeth == 'allTypes':
                            dml2 = [ [dmi[0],dmi[1],dmm2[nnzidx[idx]], dmr[idx]] for idx, dmi in enumerate(nnzidx) ]
                            idxr = sorted( dml2, key=lambda item: item[3], reverse=True)
                        elif strMeth == 'noProb':
                            idxr = random.sample( nnzidx, len(nnzidx))
                    
                    
                        # decimate matrix 
                        # one bead can only be ligated with another one
                        # alternative
                        dm2 = np.zeros( dmm2.shape) * np.nan
                        for id1, id2, dmi, pbi in idxr:
                            dm2[ id1, id2] = dmm2[ id1, id2]
                            dmm2[ id1, :] = np.nan
                            dmm2[ :, id2] = np.nan
          
                            
                        # pdb.set_trace()
                        # make symmetric again
                        dm2 = (~np.isnan(dm2)).astype(int)
                        dm2 = np.triu( dm2, k=1 ).T + dm2
                        
                        
            
                        ### accumulate counts over previous configs   
                        count[ idtv, :, :] = count[ idtv, :, :] + dm2
                
                
            
            
            nid = nid + len( ranli[2] ) 
            
        dmm = dmm / nid * dmunit
        #
        ###
        if distMode == '3.4':
            sdsk = sdsk + '_dist3.4'
        elif distMode == '3.2':
            sdsk = sdsk + '_dist3.2'
        elif distMode == '3.1.1':
            sdsk = sdsk + '_dist3.1.1'
        elif distMode == '3.1':
            pass
        else:
            sdsk = sdsk + '_dist3.0'         
        
        ####        
        countD.update( { sdsk+'_n'+str(nid)+ '_cf' + str(round(bondLengthFraction,2)) : count} )
        dmD.update( { sdsk+'_dm_n'+str(nid) : dmm} )
        
        if np.isin( distMode, ['3.2','3.4'] ):
            countD.update( { 'dftype' : dftype} )
            dmD.update( { 'dftype' : dftype} )  
        
    
    return countD, dmD














def makeCounts3( strDsd, str2model, nsampl, bondLengthFraction=2, settDictInfoFN='settDict', rseed= 42, distMode='3.1' ):
    
    exec( 'from ' + settDictInfoFN + ' import confdall', globals() )
    
    countD = {}
    dmD = {}
    
    
    ########
    dmunit = 1
    ts = 1
    
    str_sys = 'bloom2021' # brackley2016, brackley2016stiff, bloom2021
    strTimel = ['eq'] # ['condition+last','eq']
    te = 3 * 10**6 * ts
    tatol =  10 **5 * ts

    
    for sdsk, strDataset in strDsd.items():
        print( 'updating', sdsk, 'with', strDataset) 
        exec( 'import ' + str2model[ sdsk ]  + ' as model', globals() )
        
        try:
            exec( 'model = model.'+ re.sub('_model','',str2model[ sdsk ]) +'()', globals())
        except:
            pass
        
        
        dmcut = model.bond_length * bondLengthFraction
        
        nid = 0 # len( runidv) * len( tpv)
    
        ranconftimel = confChooRan2( confdall, strDataset, nsampl, rseed, strTimel)

        # loop over configs
        for idrt, ranli in enumerate( ranconftimel ):
        
            idsds = ranli[0]
            sds = strDataset[idsds]
            
            runidv = confdall[ sds ][0]
            runidstr = confdall[ sds ][1]
            tpveq = confdall[ sds ][2] # [-1,-6]
            
            if len( confdall[ sds ]) > 3 and type(confdall[ sds ][3]) is str:
                strModel = confdall[ sds ][3]
            else:
                strModel = str_sys        
    
            if type( runidstr) is list:
                readPattern = env.strStorEspr + '/'+ runidstr[0] + '/traj_' + runidstr[1] + '_%d_' + runidstr[2] + '_' +runidstr[3] + '.gz'
                printPattern = 'read conf '+str(idrt+1) + '/%d: '+ runidstr[0] + '/traj_' + runidstr[1] + '_%d_' + runidstr[2] + '_' +runidstr[3] + ' at t=%s'
            else:
                if env.strCluster == 'UbuntuHome' : 
                    readPattern = env.strStorage + '/umg/traj'+ runidstr + '%d_' + strModel+'.gz'
                elif env.strCluster == 'GDWbarbieri' :
                    readPattern = env.strStorage + '/traj'+ runidstr + '%d_' +strModel+'.gz'     
    
                printPattern = 'read conf '+str(idrt+1) + '/%d: '+ runidstr + '%d_' + strModel + ' at t=%s'

    
            # check if config exists
            boolConf = True
            while boolConf:
                try:
                    if False:
                        print('read conf ' +str(idrt+1) + '/' + str(len(ranconftimel)) + ': ' + runidstr + str( ranli[1]) + ' t'+str(ranli[2]))
                        if env.strCluster == 'UbuntuHome' : 
                            confFN = env.strStorage + '/umg/traj'+ runidstr + str( ranli[1] )+'_'+strModel+'.gz'
                        elif env.strCluster == 'GDWbarbieri' :
                            confFN = env.strStorage + '/traj'+ runidstr + str( ranli[1] )+'_'+strModel+'.gz'

                    else:
                        confFN = readPattern % ( ranli[1] )
                        print( printPattern % ( len(ranconftimel), ranli[1] , str(ranli[2]) ) )


                    #########
                    df = pd.read_csv( confFN, compression='gzip', index_col=False)
                        
                    boolConf = False
                except:
                    # pdb.set_trace()
                    print('file ' + confFN +' not found')
                   
                    # ranconftimel = confChooRan( runidv, 1, rseed, strTimel, tpveq, [ranconftimel,ranconftimel[idrt] ])
                    ranconftimel = confChooRan2( confdall, [idsds,sds] , 1, rseed, strTimel, [ranconftimel, ranli ])
                    ranli = ranconftimel[ idrt]
 
                    if len(ranli[2]) == 0: 
                        print('-> skipping to next conf')
                        break
                    else: 
                        print('-> found another conf')
                        continue
                    
            
                # loop over times
                for idtv, tmtyp in enumerate( strTimel):
                    for idt, tmei in enumerate( ranli[2]):
    
                        tmeall = df.time.unique()
                        tmev = tmeall[ tmei ]
    
                        dft = df.set_index('time').loc[ tmev].reset_index()
                        
                        # check if there's velocities
                        # if ('obs' in dft.columns) and ( dft[ dft[ 'obs'] == 'v'].shape[0] > 0 ):
                        if ('obs' in dft.columns) and np.isin( 'v', dft[ 'obs'].unique()):
                            dft = dft[ dft[ 'obs'] == 'pos']
                        
                        # check if there's duplicates
                        if len( confdall[ sds ]) > 3 and type(confdall[ sds ][3]) is int:
                            dft = dft.drop_duplicates(['x','y','z'])
                
                            
                        ####
                        if distMode == '3.4':
                            dmm2, dftype = distM3_4( dft, model.pol_type)
                        if distMode == '3.2':
                            dmm2, dftype = distM3_2( dft, model.pol_type, model.binning, model.chrstart, model.chrend, model.npol)
                        elif distMode == '3.1.1':
                            dmm2 = distM3_1p1( dft, model.pol_type, model.binning, model.chrstart, model.chrend, model.npol)
                        elif distMode == '3.1':
                            dmm2 = distM3_1( dft, model.pol_type, model.binning, model.chrstart, model.chrend, model.npol)
                        else:
                            dmm2 = distM3( dft, model.pol_type)
                            
                            
                        ###
                        if idtv == 0 and nid == 0 and idt == 0:
                            dmm = np.zeros( ( len( strTimel), dmm2.shape[0], dmm2.shape[1]) )
                            count = np.zeros( ( len( strTimel), dmm2.shape[0], dmm2.shape[1]) )
                            
                        dmm[ idtv, :, :] = dmm[ idtv, :, :] +  dmm2
                        dmm2[ dmm2 <= dmcut ] = 1
                        dmm2[ dmm2 > dmcut ] = 0
                        count[ idtv, :, :] = count[ idtv, :, :] + dmm2
            
            nid = nid + len( ranli[2] ) 
            
        dmm = dmm / nid * dmunit
        #
        ###
        if distMode == '3.4':
            sdsk = sdsk + '_dist3.4'
        elif distMode == '3.2':
            sdsk = sdsk + '_dist3.2'
        elif distMode == '3.1.1':
            sdsk = sdsk + '_dist3.1.1'
        elif distMode == '3.1':
            pass
        else:
            sdsk = sdsk + '_dist3.0'        
        
        ####
        countD.update( { sdsk+'_n'+str(nid)+ '_bl' + str(bondLengthFraction) : count} )
        dmD.update( { sdsk+'_dm_n'+str(nid)+ '_bl' + str(bondLengthFraction) : dmm} )
        
        if np.isin( distMode, ['3.2','3.4'] ):
            countD.update( { 'dftype' : dftype} )
            dmD.update( { 'dftype' : dftype} )        
    
    return countD, dmD














def makeCounts2( strDsd, str2model, nsampl, bondLengthFraction=2 ):
    
    countD = {}
    dmD = {}
    
    confdall = { 
        'short100mod': [list(range(0,20)), '' , [-1,-6] ],
        'short100orig': [list(range(20,40)), '' , [-1,-6] ] ,
        'long200origExt': [list(range(80,120)), 'Dyn', range(-1,-11,-2), 200+1000+5860 ] ,
        'long200orig': [list(range(80,120)) + list(range(971,1011)), '' , [-1,-6] ] ,
        'long100orig':[ list(range(120,160)), '' , [-1,-6] ] ,
        'short200orig':[ list(range(160,180)), '', [-1,-6] ] ,
        'long200mod': [list(range(182,282)), '', [-1,-6] ] ,
        'long100mod': [list(range(286,386)), '', [-1,-6] ] ,
        'long100/500modExt': [list(range(386,427)), 'Dyn', range(-1,-11,-3), 100+500+5860 ] ,
        'long100/500mod': [list(range(386,446)) + list(range(731,791)) + list(range(791,891)) + list(range(891,931)), '', [-1,-6] ] ,
        'long100/500mod2': [list(range(446,506)), '', [-1,-6] ] ,
        'long100/500mod3': [list(range(506,566)), '', [-1,-6] ] ,
        'long100/800mod2': [list(range(566,626)), '', [-1,-6] ] ,
        'long200/600mod2': [list(range(626,670)), '', [-1,-6] ] ,
        'long100/500mod4': [list(range(671,730)), '', [-1,-6] ] ,
        'lope3': [list(range(0,100)) + list(range(150,200)), '', range(-1,-21,-2), 'lopeDumm3' ] ,
        'lope4': [list(range(0,60)) + list(range(150,200)), '', range(-1,-21,-2), 'lopeDumm4' ] ,
             }
    
    

    
    ########
    dmunit = 1
    ts = 1
    
    str_sys = 'bloom2021' # brackley2016, brackley2016stiff, bloom2021
    strTime = ['eq'] # ['condition+last','eq']
    te = 3 * 10**6 * ts
    tatol =  10 **5 * ts
    rseed = 42
    
    for sdsk, strDataset in strDsd.items():
        print( 'updating', sdsk, 'with', strDataset) 
        exec( 'import ' + str2model[ sdsk ]  + ' as model', globals() )
        dmcut = model.bond_length * bondLengthFraction
        
        nid = 0 # len( runidv) * len( tpv)
    
        for sds in strDataset:
            runidv = confdall[ sds ][0]
            runidstr = confdall[ sds ][1]
            tpveq = confdall[ sds ][2] # [-1,-6]
            
            if len( confdall[ sds ]) > 3 and type(confdall[ sds ][3]) is str:
                strModel = confdall[ sds ][3]
            else:
                strModel = str_sys
                
                
            # choose the random sampling
            if nsampl > 0:
                random.seed( rseed)
                runidvsampl = random.sample( runidv, nsampl)
            else:
                runidvsampl = runidv    
                
            
            for idr, runi in enumerate( runidvsampl ):
                
                try:
                    print('reading config', runidstr + str(runi))
                    # df = pd.read_csv( strStorage + '/umg/traj'+ runidstr + str(runi)+'_'+strModel+'.gz', compression='gzip', index_col=False)
                    df = pd.read_csv( strStorage + '/traj'+ runidstr + str(runi)+'_'+strModel+'.gz', compression='gzip', index_col=False)
                except:
                    print('config not found. Skipping...')
                    continue
            
                for idtv, dmti in enumerate( strTime):
                    # check time condition
                    if 'condition' in re.split('\+', strTime[idtv]):
                        timev = df.time.unique()
                        tpv = np.argwhere( np.isclose( timev, te, atol=tatol ))
                        if 'last' in re.split('\+', strTime[idtv]):
                            tpv = tpv[-1]
            
                    else:
                        tpv = tpveq
                        
                
                    # print( 'selected sampling times:', timev[  tpv ] )
                
                    # pdb.set_trace()
                    # timepoints list to be checked
                    tmeall = df.time.unique()
                    tmev = tmeall[ tpv]
                    for tmei in tmev:
                        dft = df.set_index('time').loc[ tmei].reset_index()
                        
                        # check if there's velocities
                        # if ('obs' in dft.columns) and ( dft[ dft[ 'obs'] == 'v'].shape[0] > 0 ):
                        if ('obs' in dft.columns) and np.isin( 'v', dft[ 'obs'].unique()):
                            dft = dft[ dft[ 'obs'] == 'pos']
                        
                        # check if there's duplicates
                        if len( confdall[ sds ]) > 3 and type(confdall[ sds ][3]) is int:
                            dft = dft.drop_duplicates(['x','y','z'])
                            
                        # dmm2 = distM3( dft, typv)
                        dmm2 = distM3_1( dft, model.pol_type, model.binning, model.chrstart, model.chrend, model.npol)
                
                        if idtv == 0 and nid == 0:
                            dmm = np.zeros( ( len(strTime), dmm2.shape[0], dmm2.shape[1]) )
                            count = np.zeros( ( len(strTime), dmm2.shape[0], dmm2.shape[1]) )
                            
                        dmm[ idtv, :, :] = dmm[ idtv, :, :] +  dmm2
                        dmm2[ dmm2 <= dmcut ] = 1
                        dmm2[ dmm2 > dmcut ] = 0
                        count[ idtv, :, :] = count[ idtv, :, :] + dmm2
                
                nid = nid + len( tpv) 
            
        dmm = dmm / nid * dmunit
        #
        countD.update( { sdsk+'_'+str(nid) : count} )
        dmD.update( { sdsk+'_'+str(nid) : dmm} )
    
    return countD, dmD





def makeCounts( strDsd, model, nsampl ):
    countD = {}
    dmD = {}
    
    confdall = { 
        'short100mod': [list(range(0,20)), '' , [-1,-6] ],
        'short100orig': [list(range(20,40)), '' , [-1,-6] ] ,
        'long200origExt': [list(range(80,120)), 'Dyn', range(-1,-11,-2), 200+1000+5860 ] ,
        'long200orig': [list(range(80,120)) + list(range(971,1011)), '' , [-1,-6] ] ,
        'long100orig':[ list(range(120,160)), '' , [-1,-6] ] ,
        'short200orig':[ list(range(160,180)), '', [-1,-6] ] ,
        'long200mod': [list(range(182,282)), '', [-1,-6] ] ,
        'long100mod': [list(range(286,386)), '', [-1,-6] ] ,
        'long100/500modExt': [list(range(386,427)), 'Dyn', range(-1,-11,-3), 100+500+5860 ] ,
        'long100/500mod': [list(range(386,446)) + list(range(731,791)) + list(range(791,891)) + list(range(891,931)), '', [-1,-6] ] ,
        'long100/500mod2': [list(range(446,506)), '', [-1,-6] ] ,
        'long100/500mod3': [list(range(506,566)), '', [-1,-6] ] ,
        'long100/800mod2': [list(range(566,626)), '', [-1,-6] ] ,
        'long200/600mod2': [list(range(626,670)), '', [-1,-6] ] ,
        'long100/500mod4': [list(range(671,730)), '', [-1,-6] ] ,
        'lope3': [list(range(0,100)) + list(range(150,200)), '', range(-1,-21,-2), 'lopeDumm3' ] ,
        'lope4': [list(range(0,60)) + list(range(150,200)), '', range(-1,-21,-2), 'lopeDumm4' ] ,
             }
    
    
    # strDataset = ['long100/500modExt', 'long100/500mod']
    # strDataset = ['long200orig', 'long200origExt']
    # strDataset = ['lope3']
    # strDataset = ['lope4']
    
# =============================================================================
#     strDsd = {
#         'long100/500mod' : ['long100/500modExt', 'long100/500mod'] ,
#         # 'long200orig' : ['long200orig', 'long200origExt'] ,
#         #'lope3' : ['lope3'] , 
#         #'lope4' : ['lope4'] , 
#         }
#     typv = [2,3,4,5] # [2,3,4 ] , [2,3,4,5]
# =============================================================================
    # rgpol_type =  [2,3,4,5] # [2,3,4],  [2,3,4,5]
    
    
    
    
    ########
    dmunit = 1
    ts = 1
    
    str_sys = 'bloom2021' # brackley2016, brackley2016stiff, bloom2021
    strTime = ['eq'] # ['condition+last','eq']
    te = 3 * 10**6 * ts
    tatol =  10 **5 * ts
    dmcut = model.bond_length * 2
    rseed = 42
    
    for sdsk, strDataset in strDsd.items():
        print( 'updating', sdsk, 'with', strDataset)    
        nid = 0 # len( runidv) * len( tpv)
    
        for sds in strDataset:
            runidv = confdall[ sds ][0]
            runidstr = confdall[ sds ][1]
            tpveq = confdall[ sds ][2] # [-1,-6]
            
            if len( confdall[ sds ]) > 3 and type(confdall[ sds ][3]) is str:
                strModel = confdall[ sds ][3]
            else:
                strModel = str_sys
                
                
            # choose the random sampling
            if nsampl > 0:
                random.seed( rseed)
                runidvsampl = random.sample( runidv, nsampl)
            else:
                runidvsampl = runidv    
                
            
            for idr, runi in enumerate( runidvsampl ):
                
                try:
                    print('reading config', runidstr + str(runi))
                    if env.strCluster == 'UbuntuHome' : 
                        df = pd.read_csv( strStorage + '/umg/traj'+ runidstr + str(runi)+'_'+strModel+'.gz', compression='gzip', index_col=False)
                    elif env.strCluster == 'GDW' :
                        df = pd.read_csv( strStorage + '/traj'+ runidstr + str(runi)+'_'+strModel+'.gz', compression='gzip', index_col=False)
                        
                except:
                    print('config not found. Skipping...')
                    continue
            
                for idtv, dmti in enumerate( strTime):
                    # check time condition
                    if 'condition' in re.split('\+', strTime[idtv]):
                        timev = df.time.unique()
                        tpv = np.argwhere( np.isclose( timev, te, atol=tatol ))
                        if 'last' in re.split('\+', strTime[idtv]):
                            tpv = tpv[-1]
            
                    else:
                        tpv = tpveq
                        
                
                    # print( 'selected sampling times:', timev[  tpv ] )
                
                    # pdb.set_trace()
                    # timepoints list to be checked
                    tmeall = df.time.unique()
                    tmev = tmeall[ tpv]
                    for tmei in tmev:
                        dft = df.set_index('time').loc[ tmei].reset_index()
                        
                        # check if there's velocities
                        # if ('obs' in dft.columns) and ( dft[ dft[ 'obs'] == 'v'].shape[0] > 0 ):
                        if ('obs' in dft.columns) and np.isin( 'v', dft[ 'obs'].unique()):
                            dft = dft[ dft[ 'obs'] == 'pos']
                        
                        # check if there's duplicates
                        if len( confdall[ sds ]) > 3 and type(confdall[ sds ][3]) is int:
                            dft = dft.drop_duplicates(['x','y','z'])
                            
                        # dmm2 = distM3( dft, typv)
                        dmm2 = distM3_1( dft, model.pol_type, model.binning, model.chrstart, model.chrend, model.npol)
                
                        if idtv == 0 and nid == 0:
                            dmm = np.zeros( ( len(strTime), dmm2.shape[0], dmm2.shape[1]) )
                            count = np.zeros( ( len(strTime), dmm2.shape[0], dmm2.shape[1]) )
                            
                        dmm[ idtv, :, :] = dmm[ idtv, :, :] +  dmm2
                        dmm2[ dmm2 <= dmcut ] = 1
                        dmm2[ dmm2 > dmcut ] = 0
                        count[ idtv, :, :] = count[ idtv, :, :] + dmm2
                
                nid = nid + len( tpv) 
            
        dmm = dmm / nid * dmunit
        #
        countD.update( { sdsk+'_'+str(nid) : count} )
        dmD.update( { sdsk+'_'+str(nid) : dmm} )
    
    return countD, dmD



















def bed2s( bedM, bedres, strzero = True):
    
    if strzero:
        chrmin1 = bedM.chrSS1.min()
        chrmin2 = bedM.chrSS2.min()    
        
        bedM['chrSn1']= ((bedM.chrSS1 - chrmin1)/bedres).astype(int)
        bedM['chrSn2']= ((bedM.chrSS2 - chrmin2)/bedres).astype(int)

        bedDim = np.maximum( bedM['chrSn1'].max(), bedM['chrSn2'].max()) +1
    

    else:
        # choose the dimensions
        pass
    
    return bedM, bedDim
    


def bed2bed( bedM, bedres, chrstart=0 ):
    bedM['chrSS1'] = ( np.floor( ((bedM['chrStart1'] - chrstart) / bedres) ) * bedres ).astype(int)
    bedM['chrSS2'] = ( np.floor( ((bedM['chrStart2'] - chrstart) / bedres) ) * bedres ).astype(int)
    
    try:
        bedM.drop( ['chrEnd1','chrEnd2'], 1, inplace=True)
    except:
        pass
        
    return bedM



def bed2m( bedM, chrstart, bedres):
    if False:
        bedM['chrSn1'] = ((bedM['chrStart1'] - chrstart) / bedres).round().astype(int)
        bedM['chrSn2'] = ((bedM['chrStart2'] - chrstart) / bedres).round().astype(int)
    else:
        bedM['chrSn1'] = np.floor( ((bedM['chrStart1'] - chrstart) / bedres)).astype(int)
        bedM['chrSn2'] = np.floor( ((bedM['chrStart2'] - chrstart) / bedres)).astype(int)

    bedDim = np.maximum( bedM['chrSn1'].max(), bedM['chrSn2'].max()) +1

    sM = scy.sparse.csr_matrix( ( bedM['score'], ( bedM['chrSn1'], bedM['chrSn2']) ), shape = (bedDim , bedDim ))
    sM = scy.sparse.triu( sM, k=1 ).T + sM
    
    return sM, bedM
    
    





def heatMplot( M, pp, cm ):
    
    if pp == 'max':
        cass = M.nonzero()
        ppmin = np.percentile( M[cass[0],cass[1]], 0.01)
        countmin, countmax, countstep = ppmin, np.max( M), (np.max( M)-0)/100
    else:
        cass = M.nonzero()
        ppmax = np.percentile( M[cass[0],cass[1]], pp)
        ppmin = np.percentile( M[cass[0],cass[1]], 1-pp)
        countmin, countmax, countstep = ppmin, ppmax, (ppmax-0) / 100
            
    ##
    countvec = np.log( np.arange( countmin, countmax, countstep) )
    
    # check if cm is given as string
    if type( cm) is not list:
        cmv = [ chi for chi in cm]
    else :
        cmv = cm
    
    cm4 = mpl.colors.LinearSegmentedColormap.from_list('HiCcount',    cmv , N=len(countvec) -1)
    
    ##
    fig5 = plt.figure()
    widths = [1, .05]
    heights = [1 ]
    spec5 = fig5.add_gridspec(ncols=2, nrows=1, width_ratios=widths,
                              height_ratios=heights, wspace=0.00, hspace=0)
    

    adm = fig5.add_subplot(spec5[0, 0])
    acb = fig5.add_subplot(spec5[0, 1])
    
    countimg = adm.imshow( M, 
                          interpolation='none', 
                          cmap= cm4, 
                          norm = mpl.colors.LogNorm(vmin=countmin, vmax=countmax) ,
                          # vmin= countvec[0], vmax= countvec[-1] ,
                          aspect='auto', 
                          ) # terrain , viridis_r, OrRd_r, CMRmap
    cbar = fig5.colorbar( countimg, ax=acb, fraction=1 )
    # cbar.set_label('')
    # adm.invert_yaxis()
    
    
    acb.spines['right'].set_visible(False)
    acb.spines['left'].set_visible(False)
    acb.spines['top'].set_visible(False)
    acb.spines['bottom'].set_visible(False)

    # adm.axes.get_xaxis().set_visible(False)
    # adm.axes.get_yaxis().set_visible(False)
    acb.axes.get_xaxis().set_visible(False)
    acb.axes.get_yaxis().set_visible(False)
    
    adm.set_xlabel('bead position')
    adm.set_ylabel('bead position')


    return fig5














def heat2Mplot( Ml, pp, cm ):
    
    cm4 = []
    countvec = []
    normlog = []
    for idx, M in enumerate(Ml):
        if pp == 'max':
            cass = M.nonzero()
            ppmin = np.percentile( M[cass[0],cass[1]], 0.01)
            countmin, countmax, countstep = ppmin, np.max( M), (np.max( M)-0)/100
        else:
            cass = M.nonzero()
            ppmax = np.percentile( M[cass[0],cass[1]], pp)
            ppmin = np.percentile( M[cass[0],cass[1]], 1-pp)
            countmin, countmax, countstep = ppmin, ppmax, (ppmax-0) / 100
                
        ##
        countvec +=  [ np.log( np.arange( countmin, countmax, countstep) ) ]
        
        # check if cm is given as string
        if type( cm[ idx]) is not list:
            cmv = [ chi for chi in cm[idx]]
        else :
            cmv = cm[ idx]

        cm4 += [ mpl.colors.LinearSegmentedColormap.from_list('HiCcount',    cmv , N=len( countvec[-1] ) -1) ]
    
        normlog += [ mpl.colors.LogNorm(vmin=countmin, vmax=countmax) ]
    
    
    
    ##
    fig5 = plt.figure( figsize=( 8 * (len(Ml) + .2), 6 ) )
    widths = [1, .05, .3 ] * len( Ml)
    heights = [1 ]
    spec5 = fig5.add_gridspec(ncols= 3 * len(Ml), nrows=1, width_ratios=widths,
                              height_ratios=heights, wspace=0, hspace=0)
    

    for idx, M in enumerate( Ml) :
        adm = fig5.add_subplot(spec5[0, idx * (len(Ml) + 1) ])
        acb = fig5.add_subplot(spec5[0, idx * (len(Ml) + 1) + 1])
        
        counting = adm.imshow( M, 
                              interpolation='none', 
                              cmap= cm4[ idx ], 
                              norm = normlog[ idx] ,
                              # vmin= countvec[0], vmax= countvec[-1] ,
                              aspect='auto', 
                              ) # terrain , viridis_r, OrRd_r, CMRmap
        cbar = fig5.colorbar( counting, ax=acb, fraction=1 )
        # cbar.set_label('')
        # adm.invert_yaxis()
        
        
        acb.spines['right'].set_visible(False)
        acb.spines['left'].set_visible(False)
        acb.spines['top'].set_visible(False)
        acb.spines['bottom'].set_visible(False)
    
        # adm.axes.get_xaxis().set_visible(False)
        # adm.axes.get_yaxis().set_visible(False)
        acb.axes.get_xaxis().set_visible(False)
        acb.axes.get_yaxis().set_visible(False)
        
        adm.set_xlabel('bead position')
        adm.set_ylabel('bead position')


    return fig5








def heat3Mplot( Ml, pp, cm ):
    
    cm4 = []
    countvec = []
    normlog = []
    for idx, M in enumerate(Ml):
        if pp == 'max':
            cass = M.nonzero()
            ppmin = np.percentile( M[cass[0],cass[1]], 0.01)
            countmin, countmax, countstep = ppmin, np.max( M), (np.max( M)-0)/100
        else:
            cass = M.nonzero()
            ppmax = np.percentile( M[cass[0],cass[1]], pp)
            ppmin = np.percentile( M[cass[0],cass[1]], 100-pp)
            countmin, countmax, countstep = ppmin, ppmax, (ppmax-0) / 100
                
        ##
        countvec +=  [ np.log( np.arange( countmin, countmax, countstep) ) ]
        
        # check if cm is given as string
        if type( cm[ idx]) is not list:
            cmv = [ chi for chi in cm[idx]]
        else :
            cmv = cm[ idx]

        cm4 += [ mpl.colors.LinearSegmentedColormap.from_list('HiCcount',    cmv , N=len( countvec[-1] ) -1) ]
    
        normlog += [ mpl.colors.LogNorm(vmin=countmin, vmax=countmax) ]
    
    
    
    ##
    fig5 = plt.figure( figsize=( 8 * (len(Ml) + .2), 6 ) )
    widths = [1, .05, .3 ] * len( Ml)
    heights = [1 ]
    spec5 = fig5.add_gridspec(ncols= 3 * len(Ml), nrows=1, width_ratios=widths,
                              height_ratios=heights, wspace=0, hspace=0)
    

    for idx, M in enumerate( Ml) :
        adm = fig5.add_subplot(spec5[0, idx * 3 ])
        acb = fig5.add_subplot(spec5[0, idx * 3 + 1])
        
        counting = adm.imshow( M, 
                              interpolation='none', 
                              cmap= cm4[ idx ], 
                              norm = normlog[ idx] ,
                              # vmin= countvec[0], vmax= countvec[-1] ,
                              aspect='auto', 
                              ) # terrain , viridis_r, OrRd_r, CMRmap
        cbar = fig5.colorbar( counting, ax=acb, fraction=1 )
        # cbar.set_label('')
        # adm.invert_yaxis()
        
        
        acb.spines['right'].set_visible(False)
        acb.spines['left'].set_visible(False)
        acb.spines['top'].set_visible(False)
        acb.spines['bottom'].set_visible(False)
    
        # adm.axes.get_xaxis().set_visible(False)
        # adm.axes.get_yaxis().set_visible(False)
        acb.axes.get_xaxis().set_visible(False)
        acb.axes.get_yaxis().set_visible(False)
        
        adm.set_xlabel('bead position')
        adm.set_ylabel('bead position')


    return fig5






def heatMplot2( M, pp, cm, cmnorm='log' ):

    if pp == ['max']:
        cass = M.nonzero()
        ppmin = np.percentile( M[cass[0],cass[1]], 0.0001)
        mmax = np.max( M) 
        countmin, countmax, countstep = ppmin, mmax, ( mmax - 0 ) / 100
        #
        countvec = np.arange( countmin, countmax, countstep)

    elif (type(pp)== list) & (pp[0] == 'fmax' ):
        cass = M.nonzero()
        ppmin = np.percentile( M[cass[0],cass[1]], 0.0001)
        mmax = np.max( M) * pp[1]
        countmin, countmax, countstep = ppmin, mmax, (mmax - 0 ) / 100
        #
        countvec = np.arange( countmin, countmax, countstep)

    elif type(pp) == list:
        if len(pp) == 4:
            countmin, countmax, countstep, lenvec = pp
        elif len(pp) == 5:
            countmin, countmax, countstep, lenvec, strtick = pp
        countvec = [lenvec] * (lenvec)

    else:
        cass = M.nonzero()
        ppmax = np.percentile( M[cass[0],cass[1]], pp)
        ppmin = np.percentile( M[cass[0],cass[1]], 1-pp)
        countmin, countmax, countstep = ppmin, ppmax, ( ppmax - 0) / 100
        #
        countvec = np.arange( countmin, countmax, countstep)

    
    # check if cm is given as string
    if type( cm) is not list:
        cmv = [ chi for chi in cm]
    else :
        cmv = cm
    
    if cmnorm == 'log':
        colnorm = mpl.colors.LogNorm(vmin=countmin, vmax=countmax)
    elif cmnorm == 'lin':
        colnorm = mpl.colors.Normalize(vmin=countmin, vmax=countmax)
    else:
        colnorm = mpl.colors.LogNorm(vmin=countmin, vmax=countmax)
        
    #
    cm4 = mpl.colors.LinearSegmentedColormap.from_list('HiCcount',    cmv , N=len(countvec) -1)
    
    
    ##
    fig5 = plt.figure(figsize=(6,5),dpi=100)
    widths = [1, .05]
    heights = [1 ]
    spec5 = fig5.add_gridspec(ncols=2, nrows=1, width_ratios=widths,
                              height_ratios=heights, wspace=0.00, hspace=0)
    

    adm = fig5.add_subplot(spec5[0, 0])
    acb = fig5.add_subplot(spec5[0, 1])
    
    counting = adm.imshow( M, 
                          interpolation='none', 
                          cmap= cm4, 
                          norm =  colnorm,
                          # vmin= countvec[0], vmax= countvec[-1] ,
                          aspect='auto', 
                          ) # terrain , viridis_r, OrRd_r, CMRmap
    cbar = fig5.colorbar( counting, ax=acb, fraction=1 )
    # cbar.set_label('')
    # adm.invert_yaxis()
    
    
    acb.spines['right'].set_visible(False)
    acb.spines['left'].set_visible(False)
    acb.spines['top'].set_visible(False)
    acb.spines['bottom'].set_visible(False)

    # adm.axes.get_xaxis().set_visible(False)
    # adm.axes.get_yaxis().set_visible(False)
    acb.axes.get_xaxis().set_visible(False)
    acb.axes.get_yaxis().set_visible(False)
    
    adm.set_xlabel('bin position')
    adm.set_ylabel('bin position')


    return fig5, adm, acb, cbar, countvec







def hMplot2( M=[], pp=[], cm=[], typvt=[], cmnorm='log', subcc=None, plotti=None, markr='tick' ):

    if plotti is None:    
    
        if pp == ['max']:
            cass = M.nonzero()
            ppmin = np.percentile( M[cass[0],cass[1]], 0.0001)
            mmax = np.max( M) 
            countmin, countmax, countstep = ppmin, mmax, ( mmax - 0 ) / 100
            #
            countvec = np.arange( countmin, countmax, countstep)
    
        elif (type(pp)== list) & (pp[0] == 'fmax' ):
            cass = M.nonzero()
            ppmin = np.percentile( M[cass[0],cass[1]], 0.0001)
            mmax = np.max( M) * pp[1]
            countmin, countmax, countstep = ppmin, mmax, (mmax - 0 ) / 100
            #
            countvec = np.arange( countmin, countmax, countstep)
    
        elif type(pp) == list:
            if len(pp) == 4:
                countmin, countmax, countstep, lenvec = pp
            elif len(pp) == 5:
                countmin, countmax, countstep, lenvec, strtick = pp
            countvec = [lenvec] * (lenvec)
    
        else:
            cass = M.nonzero()
            ppmax = np.percentile( M[cass[0],cass[1]], pp)
            ppmin = np.percentile( M[cass[0],cass[1]], 1-pp)
            countmin, countmax, countstep = ppmin, ppmax, ( ppmax - 0) / 100
            #
            countvec = np.arange( countmin, countmax, countstep)
    
        
        # check if cm is given as string
        if type( cm) is not list:
            cmv = [ chi for chi in cm]
        else :
            cmv = cm
        
        if cmnorm == 'log':
            colnorm = mpl.colors.LogNorm(vmin=countmin, vmax=countmax)
        elif cmnorm == 'lin':
            colnorm = mpl.colors.Normalize(vmin=countmin, vmax=countmax)
        else:
            colnorm = mpl.colors.LogNorm(vmin=countmin, vmax=countmax)
            
        #
        cm4 = mpl.colors.LinearSegmentedColormap.from_list('HiCcount',    cmv , N=len(countvec) -1)
        
        
        
        if subcc is None:
            scattv = np.arange(0, len(typvt))
            dummv = np.zeros( (len(typvt)))        
            scatth = np.arange(0, len(typvt))
            dummh = np.zeros( (len(typvt)))        
            
        elif type(subcc) is dict:
            typel = subcc['types']
            dftype = subcc['dftype']
            scattv = np.argwhere( np.isin( dftype.type, typel) )
            scatth = np.argwhere( np.isin( dftype.type, typel) )
            dummv = np.zeros( (len(scattv)))        
            dummh = np.zeros( (len(scatth)))    

            typvt = np.array(typvt)[ np.int_( scattv[:,0]) ]
            
        else:
            M = M[ subcc[0][0]:subcc[0][1], subcc[1][0]:subcc[1][1]]
            scattv = np.arange(0, len(typvt))[subcc[0][0]:subcc[0][1]]
            dummv = np.zeros( (len(typvt)))[subcc[0][0]:subcc[0][1]]
            scatth = np.arange(0, len(typvt))[subcc[0][0]:subcc[0][1]]
            dummh = np.zeros( (len(typvt)))[subcc[0][0]:subcc[0][1]]
        #subi1 = :
        
        
        
            
        fig5 = plt.figure(figsize=(6,5),dpi=100)
        widths = [.02, 1, .05, .05]
        heights = [1, .02]
        spec5 = fig5.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                                  height_ratios=heights, wspace=0.00, hspace=0)
        
        adm = fig5.add_subplot(spec5[0, 1])
        absv = fig5.add_subplot(spec5[0, 0], sharey=adm)
        absh = fig5.add_subplot(spec5[1, 1], sharex=adm)
        acb = fig5.add_subplot(spec5[0, 3])
        
        counting = adm.imshow( M, 
                              interpolation='none', 
                              cmap= cm4, 
                              norm =  colnorm,
                              # vmin= countvec[0], vmax= countvec[-1] ,
                              aspect='auto', 
                              ) # terrain , viridis_r, OrRd_r, CMRmap
        cbar = fig5.colorbar( counting, ax=acb, fraction=1 )
        # cbar.set_label('')
        # adm.invert_yaxis()
        
        
        adm.axes.get_xaxis().set_visible(False)
        adm.axes.get_yaxis().set_visible(False)
        acb.axes.get_xaxis().set_visible(False)
        acb.axes.get_yaxis().set_visible(False)
        
        
        #### vertical axes
        absv.autoscale(enable=True, axis='both', tight=True)
        
        if markr == 'triangle':
            absv.scatter( dummv, scattv, s=10, c= typvt, marker='>', alpha = .2)
        elif markr == 'tick':
            # absv.scatter( dummv[ typvt!= inecol ], scattv[ typvt!= inecol ], s=40, c= typvt[ typvt!= inecol], marker='_', alpha = .2)
            absv.scatter( dummv, scattv, s=40, c= typvt, marker='_', alpha = .4)

        elif markr == 'rectangle':
            
            b = ( dummv, scattv)
            # Add rectangles
            width = .7
            height = .5
            for bidx, bb in enumerate(list(zip(*b))):
                absv.add_patch(Rectangle(
                    xy=(bb[0]-width/2, bb[1]-height/2) ,width=width, height=height,
                    linewidth=None, color=typvt[ bidx], fill=True, edgecolor=None))
            
            
        absv.set_ylim([0,M.shape[1]])
        absv.axes.get_xaxis().set_visible(False)
        absv.set_ylabel('bin position')
        absv.invert_yaxis()
        
        #### horizontal axes
        absh.autoscale(enable=True, axis='both', tight=True)
        
        if markr == 'triangle':
            absh.scatter( scatth, dummh, s=10, c= typvt, marker='^', alpha = .2)
        elif markr == 'tick':
            # absh.scatter( dummh[ typvt!= inecol ], scatth[ typvt!= inecol ], s=40, c= typvt[ typvt!= inecol], marker='|', alpha = .2)
            absh.scatter( dummv, scattv, s=40, c= typvt, marker='|', alpha = .4)
        elif markr == 'rectangle':
            
            a = ( scatth, dummh)
            # Add rectangles
            width = .5
            height = .7
            for aidx, aa in enumerate(list(zip(*a))):
                absh.add_patch(Rectangle(
                    xy=(aa[0]-width/2, aa[1]-height/2) ,width=width, height=height,
                    linewidth=None, color=typvt[ aidx], fill=True, edgecolor=None))
            
            
        absh.set_xlim([0,M.shape[0]])
        absh.axes.get_yaxis().set_visible(False)
        absh.set_xlabel('bin position')
        
        absh.spines['right'].set_visible(False)
        absh.spines['left'].set_visible(False)
        absv.spines['top'].set_visible(False)
        absv.spines['bottom'].set_visible(False)
        
        acb.spines['right'].set_visible(False)
        acb.spines['left'].set_visible(False)
        acb.spines['top'].set_visible(False)
        acb.spines['bottom'].set_visible(False)
        
        
        #fig5.set_facecolor('k')
        #absv.set_facecolor('k') #or any HTML colorcode, (R,G,B) tuple, (R,G,B,A) tuple, etcetc.
        #absh.set_facecolor('k') #or any HTML colorcode, (R,G,B) tuple, (R,G,B,A) tuple, etcetc.
        #adm.set_facecolor('k') #or any HTML colorcode, (R,G,B) tuple, (R,G,B,A) tuple, etcetc.
        #acb.set_facecolor('k') #or any HTML colorcode, (R,G,B) tuple, (R,G,B,A) tuple, etcetc.
    
    
    else:
        if subcc is None:
            pass # do not change anything       
            
        else:
            
            fig5, adm, absv, absh, acb, cbar, countvec = plotti        
            
            adm.set_xlim( subcc[0])
            absh.set_xlim( subcc[0] )
            
            adm.set_ylim( subcc[1])
            absv.set_ylim( subcc[1])        
        
                
    
    
    return fig5, adm, absv, absh, acb, cbar, countvec










def hMplot3( M=[], pp=[], cm=[], cmnorm='log', subcc=None, plotti=None, markr='tick' ):

    if plotti is None:    
        # pdb.set_trace()
        if pp == ['max']:
            if len( M) > 1:
                CR2 = M[1]
                M = M[0]
                cass = M[ (CR2 !=0) & ~np.isnan(CR2) ]
            else:
                M = M[0]
                cass = M[ M.nonzero()]
                
            ppmin = np.percentile( cass, 0.0001)
            mmax = np.max( M) 
            countmin, countmax, countstep = ppmin, mmax, ( mmax - 0 ) / 100
            #
            countvec = np.arange( countmin, countmax, countstep)
    
        elif (type(pp)== list) & (pp[0] == 'ratio' ):
            mmax = pp[1] # np.nanmax( M) 
            mmin = pp[2] # np.nanmin( M) 
            countmin, countmax, countstep = mmin, mmax, ( mmax - mmin ) / 100
            #
            countvec = np.arange( countmin, countmax, countstep)
                
    
    
        elif (type(pp)== list) & (pp[0] == 'fmax' ):
            if len( M) > 1:
                CR2 = M[1]
                M = M[0]
                cass = M[ (CR2 !=0) & ~np.isnan(CR2) ]
            else:
                M = M[0]
                cass = M[ M.nonzero()]

            ppmin = np.percentile( cass, 0.0001)
            mmax = np.max( M) * pp[1]
            countmin, countmax, countstep = ppmin, mmax, (mmax - 0 ) / 100
            #
            countvec = np.arange( countmin, countmax, countstep)
    
        elif (type(pp)== list) & (pp[0] == 'minmax' ):
            ppmin = pp[1]
            ppmax = pp[2]
            countmin, countmax, countstep = ppmin, ppmax, ( ppmax - ppmin) / 100
            #
            countvec = np.arange( countmin, countmax, countstep)            
    
            if len( M) > 1:
                CR2 = M[1]
                M = M[0]
            else:
                M = M[0]
                
        elif type(pp) == list:
            if len(pp) == 3:
                countmin, countmax, countstep = pp
            elif len(pp) == 4:
                countmin, countmax, countstep, strtick = pp
            countvec = np.arange( countmin, countmax, countstep) # [lenvec] * (lenvec)
    
            if len( M) > 1:
                CR2 = M[1]
                M = M[0]
            else:
                M = M[0]
    
        else:
            if len( M) > 1:
                CR2 = M[1]
                M = M[0]
                cass = M[ (CR2 !=0) & ~np.isnan(CR2) ]
            else:
                M = M[0]
                cass = M[ M.nonzero()]

            ppmax = np.percentile( cass, pp)
            ppmin = np.percentile( cass, 1-pp)
            countmin, countmax, countstep = ppmin, ppmax, ( ppmax - ppmin) / 100
            #
            countvec = np.arange( countmin, countmax, countstep)
    
        # build cm scaling        
        if cmnorm == 'log':
            colnorm = mpl.colors.LogNorm(vmin=countmin, vmax=countmax)
        elif cmnorm == 'lin':
            colnorm = mpl.colors.Normalize(vmin=countmin, vmax=countmax)
        else:
            colnorm = mpl.colors.LogNorm(vmin=countmin, vmax=countmax)

        # check if cm is given as string
        if type( cm) is not list:
            if cm in plt.colormaps():
                cm4 = cm
            else:
                cmv = [ chi for chi in cm]
                #
                cm4 = mpl.colors.LinearSegmentedColormap.from_list('HiCcount',    cmv , N=len(countvec) -1)
            
        else :
            cmv = cm
            #
            cm4 = mpl.colors.LinearSegmentedColormap.from_list('HiCcount',    cmv , N=len(countvec) -1)
        
        
        
        if subcc is None:
            scattv = []
            scatth = []
            maxplace = 0
            
        elif type(subcc) is dict:

            if 'dicplot' in subcc.keys():
                coltype = subcc['coltype']
                dicplot = subcc['dicplot']
                
                # dfstyle = dftype[ ['Id'] +coltype ].groupby( coltype ).agg(list)
                maxplace = 0
                scattv, scatth, dummv, dummh, markrs, colrs = [], [], [], [], [], []
                for key, val in dicplot.items():
                
                    dummv += [ np.ones( len(val['Index'])) * val['place']  ]
                    dummh += [ np.ones( len(val['Index'])) * val['place']  ]
    
                    markrs += [ val['mkr'] ]
                    colrs += [ val['rgb'] ]
    
                    # 
                    scattv += [ val['Index']]
                    scatth += [ val['Index']]    
                    
                    if maxplace < val['place']:
                        maxplace = val['place']
                    
            else:
                coltype = subcc['coltype']
                dftype = subcc['dftype']
                
                dfstyle = dftype[ ['Id'] +coltype ].groupby( coltype ).agg(list)
                
                scattv, scatth, dummv, dummh, markrs, colrs = [], [], [], [], [], []
                for imamm, mamm in dfstyle.iterrows():
                    # print( imamm[0], imamm[1], mamm.tolist()[0])
                    scvtmp = np.array(mamm.tolist())
                    schtmp = np.array(mamm.tolist())
                
                    dummv += [ np.ones( (scvtmp.shape[1])) * imamm[0]  ]
                    dummh += [ np.ones( (schtmp.shape[1])) * imamm[0]  ]
    
                    markrs += [ imamm[1] ]
                    colrs += [ imamm[2:5] ]
    
                    # 
                    scattv += [scvtmp]
                    scatth += [schtmp]
                    # labels = dfstyle.drop_duplicates(['placement','label'])[['placement','label']].values
                    
                maxplace = len(scattv)-1
            
        else:
            M = M[ subcc[0][0]:subcc[0][1], subcc[1][0]:subcc[1][1]]
            scattv = np.arange(0, len(typvt))[subcc[0][0]:subcc[0][1]]
            dummv = np.zeros( (len(typvt)))[subcc[0][0]:subcc[0][1]]
            scatth = np.arange(0, len(typvt))[subcc[0][0]:subcc[0][1]]
            dummh = np.zeros( (len(typvt)))[subcc[0][0]:subcc[0][1]]

        
        
        
            
        fig5 = plt.figure(figsize=(12,11),dpi=100)
        widths = [.06 +.005*len(scattv), 1, .05, .05]
        heights = [1, .06 + .005*len(scattv)]
        spec5 = fig5.add_gridspec(ncols=4, nrows=2, width_ratios=widths,
                                  height_ratios=heights, wspace=0.00, hspace=0)
        
        adm = fig5.add_subplot(spec5[0, 1])
        absv = fig5.add_subplot(spec5[0, 0], sharey=adm)
        absh = fig5.add_subplot(spec5[1, 1], sharex=adm)
        acb = fig5.add_subplot(spec5[0, 3])
        
        counting = adm.imshow( M, 
                              interpolation='none', 
                              cmap= cm4, 
                              norm =  colnorm,
                              # vmin= countvec[0], vmax= countvec[-1] ,
                              aspect='auto', 
                              ) # terrain , viridis_r, OrRd_r, CMRmap
        cbar = fig5.colorbar( counting, ax=acb, fraction=1 )
        # cbar.set_label('')
        # adm.invert_yaxis()
        
        
        adm.axes.get_xaxis().set_visible(False)
        adm.axes.get_yaxis().set_visible(False)
        acb.axes.get_xaxis().set_visible(False)
        acb.axes.get_yaxis().set_visible(False)
        
        
        #### vertical axes
        absv.autoscale(enable=True, axis='both', tight=True)
        
        if subcc is None:
            pass
        else:
                
            if markr == 'triangle':
    # =============================================================================
    #             for idtipi, tipi in enumerate(typel):
    #                 absv.scatter( dummv[idtipi], scattv[idtipi], s=6, c= typvtsub[idtipi], marker='>', alpha = .2)
    # =============================================================================
                for idtipi, dummi in enumerate(dummv):
                    absv.scatter( dummi, scattv[idtipi], s=20, color= colrs[idtipi], marker=markrs[idtipi],
                                 alpha = .7)
                    
                for idtipi, dummi in enumerate(dummh):
                    absh.scatter( scatth[idtipi], dummi, s=20, color= colrs[idtipi], marker=markrs[idtipi],
                                 alpha = .7)                    
                    
            elif markr == 'tick':
                # absv.scatter( dummv[ typvt!= inecol ], scattv[ typvt!= inecol ], s=40, c= typvt[ typvt!= inecol], marker='_', alpha = .2)
                absv.scatter( dummv, scattv, s=40, c= typvtsub, marker='_', alpha = .4)

                # absh.scatter( dummh[ typvt!= inecol ], scatth[ typvt!= inecol ], s=40, c= typvt[ typvt!= inecol], marker='|', alpha = .2)
                absh.scatter( dummv, scattv, s=40, c= typvtsub, marker='|', alpha = .4)                
    
            elif markr == 'rectangle':
                
                b = ( dummv, scattv)
                # Add rectangles
                width = .7
                height = .5
                for bidx, bb in enumerate(list(zip(*b))):
                    absv.add_patch(Rectangle(
                        xy=(bb[0]-width/2, bb[1]-height/2) ,width=width, height=height,
                        linewidth=None, color=typvtsub[ bidx], fill=True, edgecolor=None))
                    
                a = ( scatth, dummh)
                # Add rectangles
                width = .5
                height = .7
                for aidx, aa in enumerate(list(zip(*a))):
                    absh.add_patch(Rectangle(
                        xy=(aa[0]-width/2, aa[1]-height/2) ,width=width, height=height,
                        linewidth=None, color=typvtsub[ aidx], fill=True, edgecolor=None))                    
            
            
            
            
            
        absv.set_ylim([0,M.shape[1]])
        absv.axes.get_xaxis().set_visible(False)
        absv.set_ylabel('bin position')
        absv.invert_yaxis()
        
        #### horizontal axes
        absh.autoscale(enable=True, axis='both', tight=True)
        

            
            
        absh.set_xlim([0,M.shape[0]])
        absh.axes.get_yaxis().set_visible(False)
        absh.set_xlabel('bin position')
        
        absh.set_ylim([-.5,maxplace+.5])
        absv.set_xlim([-.5,maxplace+.5])

        absh.spines['right'].set_visible(False)
        absh.spines['left'].set_visible(False)
        absv.spines['top'].set_visible(False)
        absv.spines['bottom'].set_visible(False)
        
        acb.spines['right'].set_visible(False)
        acb.spines['left'].set_visible(False)
        acb.spines['top'].set_visible(False)
        acb.spines['bottom'].set_visible(False)
        
        
        #fig5.set_facecolor('k')
        #absv.set_facecolor('k') #or any HTML colorcode, (R,G,B) tuple, (R,G,B,A) tuple, etcetc.
        #absh.set_facecolor('k') #or any HTML colorcode, (R,G,B) tuple, (R,G,B,A) tuple, etcetc.
        #adm.set_facecolor('k') #or any HTML colorcode, (R,G,B) tuple, (R,G,B,A) tuple, etcetc.
        #acb.set_facecolor('k') #or any HTML colorcode, (R,G,B) tuple, (R,G,B,A) tuple, etcetc.
    
    
    else:
        if subcc is None:
            pass # do not change anything       
            
        else:
            
            fig5, adm, absv, absh, acb, cbar, countvec = plotti        
            
            adm.set_xlim( subcc[0])
            absh.set_xlim( subcc[0] )
            
            adm.set_ylim( subcc[1])
            absv.set_ylim( subcc[1])        
        
                
    
    
    return fig5, adm, absv, absh, acb, cbar, countvec









# =============================================================================
# Check functions
# =============================================================================



def check_lopPairs( lopePairs, sysanis, model):

    lpdf = pd.DataFrame( lopePairs)
    lpdf.columns = ['poltype', 'anistype', 'strand']
    lpdf.sort_values(by='anistype', inplace=True)
    
    boolLopairs = False
    
    lloop = lpdf.merge( sysanis, how= 'left', left_on = 'anistype', right_on = 'id' )
    # this filters only pairs belonging to the same dimer
    llooptmp = lloop.groupby('dimPairs').agg( len).reset_index()[['dimPairs','id']]
    llooptmp = llooptmp[ llooptmp ['id'] == 2 ]
    # llooptmp.columns = ['dimPairs','id']
    lloop2 = lloop.merge( llooptmp['dimPairs'], how='right', on='dimPairs' )
    # this filters only for dimers binding locally, not binding to distal sites
    if lloop2.shape[0] > 0 :
        lloop3 = lloop2.pivot_table( columns = 'type', index='dimPairs', values='poltype')
        lloop3['loopDim'] = lloop3[ model.anistype[0]] - lloop3[ model.anistype[1]]
        
        ### this filters for local loops (loopDim=1,2), but also well directioned (loopDim!=-1,-2)
        loopid1 = lloop3[ np.isin( lloop3.loopDim, model.minLoopSize ) ].reset_index()[ model.anistype[0] ].tolist() + \
                    lloop3[ np.isin( lloop3.loopDim, model.minLoopSize ) ].reset_index()[ model.anistype[1] ].tolist()
        lopePairs = lopePairs[ np.isin( lopePairs[:,0], loopid1), : ]      
    
    
        boolLopairs = (lloop3.loopDim < 0).sum() > 0
    
    
    return boolLopairs



def reportLoops( lopePairs, sysanis, model):

    lpdf = pd.DataFrame( lopePairs)
    lpdf.columns = ['poltype', 'anistype', 'strand']
    lpdf.sort_values(by='anistype', inplace=True)
    
    boolLopairs = False
    
    lloop = lpdf.merge( sysanis, how= 'left', left_on = 'anistype', right_on = 'id' )
    # this filters only pairs belonging to the same dimer
    llooptmp = lloop.groupby('dimPairs').agg( len).reset_index()[['dimPairs','id']]
    llooptmp = llooptmp[ llooptmp ['id'] == 2 ]
    # llooptmp.columns = ['dimPairs','id']
    lloop2 = lloop.merge( llooptmp['dimPairs'], how='right', on='dimPairs' )
    # this filters only for dimers binding locally, not binding to distal sites
    if lloop2.shape[0] > 0 :
        lloop3 = lloop2.pivot_table( columns = 'type', index='dimPairs', values='poltype')
        lloop3['loopDim'] = lloop3[ model.anistype[0]] - lloop3[ model.anistype[1]]
        
        ### this filters for local loops (loopDim=1,2), but also well directioned (loopDim!=-1,-2)
        loopid1 = lloop3[ np.isin( lloop3.loopDim, model.minLoopSize ) ].reset_index()[ model.anistype[0] ].tolist() + \
                    lloop3[ np.isin( lloop3.loopDim, model.minLoopSize ) ].reset_index()[ model.anistype[1] ].tolist()
        lopePairs = lopePairs[ np.isin( lopePairs[:,0], loopid1), : ]      
    
    
        #boolLopairs = (lloop3.loopDim < 0).sum() > 0
    
    else:
        lloop3 = np.empty((0,3), dtype=np.int64)
    
    return lloop3
















def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print( 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj) )













 
def createAnnot( model ):

    dftidx = model.binnffill[model.binnffill.Index >= 0][['Index','annot','strand']]

    annotIndx = dftidx[['annot','Index','strand']].drop_duplicates()
    
    annotIndx.Index = annotIndx.Index * model.bpres / 1000
    ctcfdixn = np.isin( annotIndx.annot , ['CTCF','CTCF+GENE']) & np.isin( annotIndx.strand , ['+'])
    ctcfdixp = np.isin( annotIndx.annot , ['CTCF','CTCF+GENE']) & np.isin( annotIndx.strand , ['-'])
    promdixn = np.isin( annotIndx.annot , ['PROMOTER']) & np.isin( annotIndx.strand , ['+'])
    promdixp = np.isin( annotIndx.annot , ['PROMOTER']) & np.isin( annotIndx.strand , ['-'])
    enhdix = np.isin( annotIndx.annot , ['ENHANCER'])
    tesdix = np.isin( annotIndx.annot , ['TES'])
    
    annotIndx['tickLabel'] = 'NaN'
    annotIndx.tickLabel[ ctcfdixp] = '\n>'
    annotIndx.tickLabel[ ctcfdixn] = '\n<'
    annotIndx.tickLabel[ promdixp] = '<'
    annotIndx.tickLabel[ promdixn] = '>'
    annotIndx.tickLabel[ enhdix] = '^'
    annotIndx.tickLabel[ tesdix] = '|'
    annotIndx2 = annotIndx[annotIndx.tickLabel != 'NaN']             
    

    # add other annotations : ALL NOTABLE ELEMENTS
    annotIndx['annot2'] = 'int'
    annotIndx['annot2'][ctcfdixp] = [ '>'+str(anncoo) for anncoo in  range(annotIndx[ctcfdixp].shape[0]) ] 
    annotIndx['annot2'][ctcfdixn] = [ '<'+str(anncoo) for anncoo in  range(annotIndx[ctcfdixn].shape[0]) ] 
    annotIndx['annot2'][promdixp] = [ 'p<'+str(anncoo) for anncoo in  range(annotIndx[promdixp].shape[0]) ] 
    annotIndx['annot2'][promdixn] = [ 'p>'+str(anncoo) for anncoo in  range(annotIndx[promdixn].shape[0]) ] 
    annotIndx['annot2'][enhdix] = [ '^'+str(anncoo) for anncoo in  range(annotIndx[enhdix].shape[0]) ] 
    annotIndx['annot2'][tesdix] = [ '|'+str(anncoo) for anncoo in  range(annotIndx[tesdix].shape[0]) ] 


    # add other annotations : FISH PROBES
    chrArt = pd.read_excel( model.artifSys[0] , engine="odf", sheet_name= model.artifSys[1], index_col=False)
    chrArt2 = chrArt[~chrArt['chrStart from bed'].isna()][['chromStart','chrStart from bed','probe name','fish']]
    # chrArt2.iloc[[0,-1]]
    
    mcoo = (chrArt2.iloc[-1]['chrStart from bed'] - chrArt2.iloc[0]['chrStart from bed'])/(chrArt2.iloc[-1].chromStart - chrArt2.iloc[0].chromStart)
    qcoo = chrArt2.iloc[0]['chrStart from bed'] - mcoo * (chrArt2.iloc[0].chromStart + model.taillen )
    
    
    chrprobes = ((chrArt.fish -  qcoo) / mcoo ).round()
    chrArt2['probe position'] = chrprobes
    chrArt2 = chrArt2[~chrArt2['probe position'].isna()]
    

    # annotIndx['annot3'] = 'int'
    annotIndx = annotIndx.merge( chrArt2, right_on='probe position', left_on='Index', how='left')
    annotIndx['annot3'] = annotIndx['probe name']    
    
    
    return annotIndx
    






