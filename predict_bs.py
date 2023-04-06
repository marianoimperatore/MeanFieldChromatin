#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:52:49 2022

@author: mariano
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


import esprenv as env
import func0 as func
import funinv as funinv
import setInvDict as invdict


# =============================================================================
# General settings
# =============================================================================
strSampl = '148' # 394B, 148

    
# extract the sub tracks from the sub region
bpres = 5000
subreg = [57.7*10**6, 58.155*10**6]
extreg = [57.66*10**6, 58.33*10**6]

chrrange1 = ['chr12',int(57.7*10**6),int(58.155*10**6),bpres]
chrrange2 = ['chr8',int(127.71*10**6),int(127.78*10**6),5000]


startsubid = int((subreg[0]-extreg[0])/bpres)
endsubid = int((subreg[1]-extreg[0])/bpres)
    
    


# =============================================================================
# SA setting
# =============================================================================
N = 134
M = 12

if strSampl == '148':
        
    strRegg = 'GSC275B+C88bex2+5000'
    strRegg2 = 'GSC148'
    
elif strSampl == '394B':

    strRegg = 'GSC394B+C88bex2+5000'
    strRegg2 = 'GSC394B'
    

svfile = env.strData + '/SV_mapOrient_GSC'+strSampl+'+C88bex2+5000_M'+str(M)



# =============================================================================
# Read tracks for estimation of probability normalization
# =============================================================================
# 148
K27ac21 = np.genfromtxt( env.strData + '_'.join([ 'h3k27+inferenceRegion+avg', 'GSC148',str(bpres) ])+'.csv', delimiter=',' )
RNAseq21 = np.genfromtxt( env.strData + '_'.join([ 'rnaseq+inferenceRegion+avg', 'GSC148',str(bpres) ])+'.csv' , delimiter=',' )
K27ac2P1 = np.genfromtxt( env.strData + '_'.join([ 'h3k27+predictionRegion+avg', 'GSC148',str(bpres) ])+'.csv', delimiter=',' )
RNAseq2P1 = np.genfromtxt( env.strData + '_'.join([ 'rnaseq+predictionRegion+avg', 'GSC148',str(bpres) ])+'.csv', delimiter=',' )
# 394B
K27ac22 = np.genfromtxt( env.strData + '_'.join([ 'h3k27+inferenceRegion+avg', 'GSC394B',str(bpres) ])+'.csv', delimiter=',' )
RNAseq22 = np.genfromtxt( env.strData + '_'.join([ 'rnaseq+inferenceRegion+avg', 'GSC394B',str(bpres) ])+'.csv' , delimiter=',' )
K27ac2P2 = np.genfromtxt( env.strData + '_'.join([ 'h3k27+predictionRegion+avg', 'GSC394B',str(bpres) ])+'.csv', delimiter=',' )
RNAseq2P2 = np.genfromtxt( env.strData + '_'.join([ 'rnaseq+predictionRegion+avg', 'GSC394B',str(bpres) ])+'.csv', delimiter=',' )


maxchrrefsamptargrseq= np.maximum( 
                        np.maximum( RNAseq21.max(),RNAseq2P1.max()),
                        np.maximum( RNAseq22.max(),RNAseq2P2.max())
                                   )
maxchrrefsamptargk27 = np.maximum( 
                        np.maximum( K27ac21.max(),K27ac2P1.max()),
                        np.maximum( K27ac22.max(),K27ac2P2.max())
                                   )









# =============================================================================
# Load tracks for binding sites inference and prediction
# =============================================================================
K27ac2 = np.genfromtxt( env.strData + '_'.join([ 'h3k27+inferenceRegion+avg', strRegg2,str(bpres) ])+'.csv', delimiter=',' )
RNAseq2 = np.genfromtxt( env.strData + '_'.join([ 'rnaseq+inferenceRegion+avg',strRegg2,str(bpres) ])+'.csv' , delimiter=',' )
K27acP2 = np.genfromtxt( env.strData + '_'.join([ 'h3k27+predictionRegion+avg',strRegg2,str(bpres) ])+'.csv', delimiter=',' )
RNAseqP2 = np.genfromtxt( env.strData + '_'.join([ 'rnaseq+predictionRegion+avg', strRegg2,str(bpres) ])+'.csv', delimiter=',' )








# =============================================================================
# # correlations
# =============================================================================


fn = '_'.join([ env.strData + 'bestInference', strRegg, 'N'+str(N), 'M'+str(M) ]) + '.csv'
beststar = pd.read_csv(fn,  sep = ',', index_col=False)    
##
R= beststar.R.values[0]
M= beststar.M.values[0]
N= beststar.N.values[0]
NR = N*R
strMV = 'cf' + str(beststar.cfmax.values[0])

ttcsel = beststar[ [ str(ni) for ni in range(0, NR) ] ].values.flatten()        

Ri = np.arange( 0, ttcsel[startsubid * R:endsubid * R].size, R, dtype=np.int_)

###






### extended region
Rix = np.arange( 0, ttcsel.size, R, dtype=np.int_)

tracksExt = []
Mv = []
for mi in range(1, M):
    tracki = np.add.reduceat( np.int_(ttcsel == mi), Rix)/R
    if np.all(tracki==0): 
        print(mi)
        continue

    Mv += [mi]
    tracksExt += [ tracki]
    

Ms = len(Mv)


                
##
fig, ax = plt.subplots(
    nrows= 1,
    ncols= 1, 
    figsize=(12,2)
    )
cmap = plt.cm.get_cmap('tab20', Ms)
for trackid, tracksi in enumerate(tracksExt[::-1]):
    ax.fill_between(range(0,len(tracksi)),tracksi, 
                    color=cmap( ((len(tracksExt)-1 -trackid)/(Ms-1)) % 1.1),
                    alpha=.5
                    )
    ax.set_xlim([0,len(tracksi)])
    
ax.set_ylim([0,1])
ax.set_yticks([0,.5,1])
ax.set_yticklabels(['0','0.5','1'])
fig.tight_layout()


plt.savefig( env.strHome + '_'.join([ '/track+ref+ext_dense', strRegg, 'N'+str(N), 'M'+str(M) ])+'.svg', 
            format='svg', dpi=500 ) 





###
tracks = []
Mt= 1
Mvref = []
for mi in Mv:
    tracki = np.add.reduceat( np.int_(ttcsel[startsubid * R:endsubid * R] == mi), Ri)/R
    if np.all(tracki==0): 
        print(mi)
        Mvref += [np.nan]
    else:
        Mvref += [Mt]
        Mt +=1

    tracks += [ tracki]
                
t2c2 = pd.DataFrame(data={
    'type': Mv,
    'class': Mvref,
    })

# plot tracks
type2color = pd.DataFrame([])
t2c = pd.DataFrame([])

fig, ax = plt.subplots(
    nrows= len(tracks),
    ncols= 1, 
    figsize=(15,len(tracks))
    )
cmap = plt.cm.get_cmap('tab20', Ms)
for trackid, tracksi in enumerate(tracks[::-1]):
    colri = cmap( ((len(tracks)-1 -trackid)/(Ms-1)) % 1.1)
   
    ax[trackid].fill_between(range(0,len(tracksi)),tracksi, color=colri)
    ax[trackid].set_xlim([0,len(tracksi)])
    ax[trackid].set_ylim([0,1])
    ax[trackid].set_ylabel(len(tracks)-trackid)
    if trackid != len(tracks):
        ax[trackid].axes.get_xaxis().set_visible(False)
    
    
    t2c = t2c.append(
        pd.DataFrame(data={
        'color':[','.join([str(collri) for collri in colri])],
        'type':[len(tracks)-trackid],
        }))

##    
t2c = t2c.merge(t2c2, on='type', how='left')
##  
t2c['mode'] = '_'.join([ 'track+ref_full', strRegg, 'N'+str(N), 'M'+str(M) ])
type2color = type2color.append( t2c )
    



##  
fig.tight_layout()
plt.savefig( env.strHome +'_'.join([ '/track+ref_full', strRegg, 'N'+str(N), 'M'+str(M) ])+'.svg', 
            format='svg', dpi=500 ) 



###
fig, ax = plt.subplots(
    nrows= 1,
    ncols= 1, 
    figsize=(12,2)
    )
cmap = plt.cm.get_cmap('tab20', Ms)
for trackid, tracksi in enumerate(tracks[::-1]):
    ax.fill_between(range(0,len(tracksi)),tracksi, 
                    color=cmap( ((len(tracks)-1 -trackid)/(Ms-1)) % 1.1),
                    alpha=.5
                    )
    ax.set_xlim([0,len(tracksi)])
    
ax.set_ylim([0,1])
ax.set_yticks([0,.5,1])
ax.set_yticklabels(['0','0.5','1'])
fig.tight_layout()

plt.savefig( env.strHome +'_'.join([ '/track+ref_dense', strRegg, 'N'+str(N), 'M'+str(M) ])+'.svg', 
            format='svg', dpi=500 ) 












# =============================================================================
# Evaluate correlations
# =============================================================================
classes = pd.DataFrame(tracks).T
classes.columns = classes.columns+1
classes['k27ac'] = K27ac2/maxchrrefsamptargk27
classes['rnaseq'] = RNAseq2/maxchrrefsamptargrseq
classcorr = classes.corr(method='pearson')

classcorr.to_csv(
    env.strHome + '_'.join([ '/classcorr', strRegg, 'N'+str(N), 'M'+str(M) ])+'.csv')
    



# =============================================================================
# ## calculate conditional probabilities
# =============================================================================
Probs = classes.sum(0) / classes.shape[0]
probprod = Probs * (1-Probs)

def probCorr2( A, B, probprod, Probs, classcorr):
    probsqrt1 = np.sqrt( probprod[A] * probprod[B])
    return Probs[A] + classcorr[B][A] * probsqrt1 / Probs[B]


colors = list(range(1,Ms+1))
trakis = ['k27ac','rnaseq']
condprob = classcorr * 0
for coli in colors:
    for traki in trakis:    
        if classcorr[coli][traki] <= 0.195: continue
    
        condprob[coli][traki] = probCorr2( coli, traki, probprod, Probs, classcorr)


condprob.to_csv(
    env.strHome + '_'.join([ '/condprob', strRegg, 'N'+str(N), 'M'+str(M) ])+'.csv')
    


########
condprob2 = condprob.T.iloc[:-2][['k27ac','rnaseq']]
condprob2['type'] = condprob2.index
type2color = type2color.merge(condprob2, on='type',how='left')

listVMDname = ['A','B','C','D','E','F','G']
type2color['vmdname'] = ''
pdSV = pd.read_csv( svfile + '.csv')
pdSVidx = pdSV.groupby(['chr'], sort=False).agg(len).reset_index()[['Idx']]


t2cmask = ((type2color[['k27ac','rnaseq']] != 0) & (~type2color[['k27ac','rnaseq']].isna())).any(1)
type2color['vmdname'][t2cmask] = listVMDname[ pdSVidx.shape[0] : pdSVidx.shape[0]+ t2cmask.sum()][::-1]


type2color.to_csv(
    env.strHome +'_'.join([ 'typ2colr', strRegg, 'N'+str(N), 'M'+str(M) ])+ '.csv'
    , index=False) 






######
corrmap = classcorr.iloc[:-2][['k27ac','rnaseq']]
corrmap['class'] = corrmap.index.astype(int)
cmap = plt.cm.get_cmap('tab20', Ms)

fig, ax = plt.subplots(figsize=(2.5,7))
ax = sns.heatmap( corrmap[['k27ac','rnaseq']], annot=True, fmt=".2f", vmin=-1,vmax=1,cmap='bwr')
ax.xaxis.tick_top()
ax.get_yaxis().set_visible(False)
fig.tight_layout()

fig, ax = plt.subplots(figsize=(1,7))
ax = sns.heatmap( corrmap[['class']], annot=True, cmap=cmap, cbar=False)
ax.xaxis.tick_top()
ax.get_yaxis().set_visible(False)
fig.tight_layout()
        

    
###
cmap2 = cmap.colors[(~corrmap[['k27ac']].isna()).values.flatten()]
corrmap = classcorr.dropna(how='all',axis=0).iloc[:-2][['k27ac','rnaseq']]
corrmap.index = corrmap.reset_index().index +1
corrmap['class'] = corrmap.index.astype(int)

fig, ax = plt.subplots(figsize=(2.5,7))
ax = sns.heatmap( corrmap[['k27ac','rnaseq']], annot=True, fmt=".2f", vmin=-1,vmax=1,cmap='bwr')
ax.xaxis.tick_top()
ax.get_yaxis().set_visible(False)
fig.tight_layout()
plt.savefig( env.strHome +'_'.join([ '/corrmap1', strRegg, 'N'+str(N), 'M'+str(M) ])+'.svg', 
            format='svg' ) 

fig, ax = plt.subplots(figsize=(1,7))
ax = sns.heatmap( corrmap[['class']], annot=True, cmap=matplotlib.colors.ListedColormap(cmap2), cbar=False)
ax.xaxis.tick_top()
ax.get_yaxis().set_visible(False)
fig.tight_layout()
plt.savefig( env.strHome + '_'.join([ '/corrmap2', strRegg, 'N'+str(N), 'M'+str(M) ])+'.svg', 
            format='svg' ) 
            


corrmap.to_csv( env.strHome + '_'.join([ '/corrmap', strRegg, 'N'+str(N), 'M'+str(M) ])+'.csv'
               , index=False)

# =============================================================================
# Predictions
# =============================================================================


#### prob
K27acP2n = K27acP2 / maxchrrefsamptargk27
RNAseqP2n = RNAseqP2 / maxchrrefsamptargrseq

trackspred = pd.DataFrame(data={
    'k27ac': K27acP2n,
    'rnaseq': RNAseqP2n}
    )
pabc = pd.DataFrame( np.zeros((trackspred.shape[0],len(colors))) )
pabc.columns = colors
for coli in colors:
    for traki in trakis:
        pabc[coli] = pabc[coli] + condprob[coli][traki] * trackspred[traki]
    

pabcargmax = (pabc.sum(1)>1)
pabcmax = pabc[pabcargmax].sum(1)
pabc2 = pabc.copy()
pabc2[ pabcargmax] = pabc2[ pabcargmax] / pabcmax


pabcR = pd.DataFrame(np.repeat( pabc2.values, R, axis=0))
pabcR.columns= colors

pp = pd.concat((1-pabcR.sum(1), pabcR),1)
pp[pp.isna()] = 0

tenti = 100
ttpred = np.zeros((tenti,pp.shape[0]), dtype=int)
for bpid, pabci in pp.iterrows():
    ttpred[:, bpid] = np.random.choice( list(range(0,Ms+1)), tenti, p=pabci)


pd.DataFrame(ttpred).to_csv( env.strHome +'_'.join([ '/SV_ttc', strRegg, 'N'+str(N), 'M'+str(M) ])+'.csv'
                            , index=False)
pd.DataFrame(ttpred).to_csv( env.strHome +'_'.join([ '/SV_ttc', strRegg, 'N'+str(N), 'M'+str(M) ])+'.csv'
                            , index=False)

pd.DataFrame(pp).to_csv( env.strHome +'_'.join([ '/predictedbs', strRegg, 'N'+str(N), 'M'+str(M) ])+'.csv'
                            , index=False)


### average tracks
RiPred = np.arange( 0, ttpred.shape[1], R, dtype=np.int_)
trackPred = []
# for mi in range(1, Ms):
for mi in range(1, Ms+1):
    tracktemp = np.zeros(pabc.shape[0])
    for tti in range(0, tenti):
        tracktemp += np.add.reduceat( np.int_(ttpred[tti,:] == mi), RiPred)/R
    trackPred += [ tracktemp/ tenti]



# plot trackPred + original tracks

# extract the sub tracks from the sub region
bpres = 5000
subreg = [57.7*10**6, 58.155*10**6]
extreg = [57.66*10**6, 58.33*10**6]

startsubid = int((subreg[0]-extreg[0])/bpres)
endsubid = int((subreg[1]-extreg[0])/bpres)


###
trackAll = []
for mi, tracki in enumerate(tracks):
    trackAll += [ np.concatenate( (trackPred[mi], tracks[mi])) ]


fig, ax = plt.subplots(
    nrows= len(trackAll),
    ncols= 1, 
    figsize=(15,len(trackAll))
    )
cmap = plt.cm.get_cmap('tab20', Ms)
for trackid, tracksi in enumerate(trackAll[::-1]):
    ax[trackid].fill_between(range(0,len(tracksi)),tracksi, color=cmap( ((len(trackAll)-1 -trackid)/(M-2)) % 1.1))
    ax[trackid].set_xlim([0,len(tracksi)])
    ax[trackid].set_ylim([0,1])
    ax[trackid].set_ylabel(len(trackAll)-trackid)
    if trackid != len(trackAll)-1:
        ax[trackid].axes.get_xaxis().set_visible(False)
    
fig.tight_layout()
plt.savefig( env.strHome + '_'.join([ '/track+pred_full', strRegg, 'N'+str(N), 'M'+str(M) ])+'.svg', 
            format='svg' ) 


fig, ax = plt.subplots(
    nrows= 1,
    ncols= 1, 
    figsize=(12,2)
    )
cmap = plt.cm.get_cmap('tab20', Ms)
for trackid, tracksi in enumerate(trackAll[::-1]):
    ax.fill_between(range(0,len(tracksi)),tracksi, 
                    # color=cmap( ((len(trackAll)-1 -trackid)/(Ms-2)) % 1.1),
                    color=cmap( ((len(trackAll)-1 -trackid)/(Ms-1)) % 1.1),
                    alpha=.5
                    )
    ax.set_xlim([0,len(tracksi)])
    
ax.set_ylim([0,1])
ax.set_yticks([0,.5,1])
fig.tight_layout()

plt.savefig( env.strHome + '_'.join([ '/track+pred_dense', strRegg, 'N'+str(N), 'M'+str(M) ])+'.svg', 
            format='svg' ) 



# =============================================================================
# # make plot of new matrix
# =============================================================================


ttsub = ttcsel[startsubid * R:endsubid * R]

ttall = np.concatenate( ( ttpred,
                     np.repeat(ttsub[:,None], tenti, axis=1).T
                     ), axis=1)

NRp = ttall.shape[1]

diaggip = np.diag_indices( NRp, ndim=2) 
sijp = np.zeros((NRp,NRp), dtype=np.int_)

for digi in range(0,NRp):
    sijp[ (diaggip[0]+digi) % NRp, diaggip[1] ] = digi

sijp = np.tril( sijp )+ np.tril(sijp).T

Rip = np.arange( 0, ttall.shape[1], R, dtype=np.int_)



##########
def buildP( bb, ppc, ppo, Ri):
    ppi = np.where( (bb[None,:] == bb[:, None]) & (bb[None,:] != 0) & (bb[:,None] != 0), ppc, ppo )
    ppi = np.where( (bb[None,:] != bb[:, None]) & (bb[None,:] != 0) & (bb[:,None] != 0), 0, ppi )
    
    ppir = np.add.reduceat( np.add.reduceat( ppi, Ri), Ri, 1)    
   
    return ppir 




#####
ppop, ppcp = funinv.buildScaling( NRp, beststar.alpscal.iloc[0], sijp, strPh='interpolate')
ttcop = np.zeros((NRp))
Po = buildP(ttcop, ppcp, ppop, Rip)

# 
Piavg = np.zeros((int(ttall.shape[1]/R),int(ttall.shape[1]/R)))
for tti in range(tenti):
    Piavg += buildP(ttall[tti,:], ppcp, ppop, Rip)
Piavg /= tenti


# plot and save matrix
alp = beststar.alp.values[0]
funinv.plRes2( alp* Piavg + (1- alp) * Po, trackAll,  ppv=[70,100], figsiz=(6,9))    

plt.savefig( env.strHome +'_'.join([ '/cm+sv_pred', strRegg, 'N'+str(N), 'M'+str(M) ])+'.svg', 
            format='svg', dpi=500 ) 



pd.DataFrame(alp* Piavg + (1- alp) * Po).to_csv(env.strHome +'_'.join([ '/cm+sv_pred', strRegg, 'N'+str(N), 'M'+str(M) ])+'.csv'
                                                , index=False)





