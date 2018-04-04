# Copyright (c) 2017 Johan Berglund
# BlochBuster is distributed under the terms of the GNU General Public License
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import proj3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.patches import FancyArrowPatch
import numpy as np
import scipy.integrate as integrate
import os.path
import shutil
import optparse
import yaml
import FFMPEGwriter


colors = {  'bg':       [1,1,1], 
            'circle':   [0,0,0,.03],
            'axis':     [.5,.5,.5],
            'text':     [.05,.05,.05], 
            'spoilText':[.5,0,0],
            'RFtext':   [0,.5,0],
            'Gtext':    [80/256,80/256,0],
            'comps': [  [.3,.5,.2],
                        [.1,.4,.5],
                        [.5,.3,.2],
                        [.5,.4,.1],
                        [.4,.1,.5],
                        [.6,.1,.3]],
            'boards': { 'B1': [.5,0,0],
                        'Gx': [0,.5,0],
                        'Gy': [0,.5,0],
                        'Gz': [0,.5,0]
                        },
            'kSpacePos': [1, .5, 0]
            }

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


# Creates an animated plot of magnetization in a 3D view
def plotFrame3D(config, vectors, frame):
    nx, ny, nz, nComps, nIsoc = vectors.shape[:5]
    xpos = np.arange(nx)-nx/2+.5
    ypos = -(np.arange(ny)-ny/2+.5)
    zpos = -(np.arange(nz)-nz/2+.5)

    # Create 3D axes
    if nx*ny*nz==1:
        aspect = .95 # figure aspect ratio
    elif nz==1 and ny==1 and nx>1:
        aspect = 0.6
    elif nz==1 and nx>1 and ny>1:
        aspect = .75
    else:
        aspect = 1
    figSize = 5 # figure size in inches
    canvasWidth = figSize
    canvasHeight = figSize*aspect
    fig = plt.figure(figsize=(canvasWidth, canvasHeight))
    axLimit = max(nx,ny,nz)/2+.5
    ax = fig.gca(projection='3d', xlim=(-axLimit,axLimit), ylim=(-axLimit,axLimit), zlim=(-axLimit,axLimit), fc=colors['bg'])
    ax.set_aspect('equal')
    
    if nx*ny*nz>1:
        azim = -78 # azimuthal angle of x-y-plane
        ax.view_init(azim=azim) #ax.view_init(azim=azim, elev=elev)
    ax.set_axis_off()
    width = 1.65 # to get tight cropping
    height = width/aspect
    left = (1-width)/2
    bottom = (1-height)/2
    if nx*ny==1: # shift to fit legend
        left += .035
        bottom += -.075
    else:
        bottom += -.085
    ax.set_position([left, bottom, width, height])

    if nx*ny*nz == 1:
        # Draw axes circles
        for i in ["x", "y", "z"]:
            circle = Circle((0, 0), 1, fill=True, lw=1, fc=colors['circle'])
            ax.add_patch(circle)
            art3d.pathpatch_2d_to_3d(circle, z=0, zdir=i)

        # Draw x, y, and z axes
        ax.plot([-1, 1], [0, 0], [0, 0], c=colors['axis'], zorder=-1)  # x-axis
        ax.text(1.08, 0, 0, r'$x^\prime$', horizontalalignment='center', color=colors['text'])
        ax.plot([0, 0], [-1, 1], [0, 0], c=colors['axis'], zorder=-1)  # y-axis
        ax.text(0, 1.12, 0, r'$y^\prime$', horizontalalignment='center', color=colors['text'])
        ax.plot([0, 0], [0, 0], [-1, 1], c=colors['axis'], zorder=-1)  # z-axis
        ax.text(0, 0, 1.05, r'$z$', horizontalalignment='center', color=colors['text'])

    # Draw title:
    fig.text(.5, 1, config['title'], fontsize=14, horizontalalignment='center', verticalalignment='top', color=colors['text'])

    # Draw time
    time_text = fig.text(0, 0, 'time = %.1f msec' % (config['clock'][frame%(len(config['clock'])-1)]), color=colors['text'], verticalalignment='bottom')

    # TODO: put isochromats in this order from start
    order = [int((nIsoc-1)/2-abs(m-(nIsoc-1)/2)) for m in range(nIsoc)]
    # Draw magnetization vectors
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                for c in range(nComps):
                    for m in range(nIsoc):
                        col = colors['comps'][(c) % len(colors['comps'])]
                        M = vectors[x,y,z,c,m,:,frame]
                        Mnorm = np.linalg.norm(M)
                        alpha = 1.-2*np.abs((m+.5)/nIsoc-.5)
                        if Mnorm>.05:
                            ax.add_artist(Arrow3D(  [xpos[x], xpos[x]+M[0]], 
                                                    [ypos[y], ypos[y]+M[1]],
                                                    [zpos[z], zpos[z]+M[2]], 
                                                    mutation_scale=20,
                                                    arrowstyle="-|>", lw=2,
                                                    color=col, alpha=alpha, 
                                                    zorder=order[m]+nIsoc*int(100*(1-Mnorm))))
        
    # Draw "spoiler" and "FA-pulse" text
    fig.text(1, .94, config['RFtext'][frame], fontsize=14, alpha=config['RFalpha'][frame],
            color=colors['RFtext'], horizontalalignment='right', verticalalignment='top')
    fig.text(1, .88, config['Gtext'][frame], fontsize=14, alpha=config['Galpha'][frame],
            color=colors['Gtext'], horizontalalignment='right', verticalalignment='top')
    fig.text(1, .82, 'spoiler', fontsize=14, alpha=config['spoilAlpha'][frame],
            color=colors['spoilText'], horizontalalignment='right', verticalalignment='top')
    
    # Draw legend:
    for c in range(nComps):
        col = colors['comps'][(c) % len(colors['comps'])]
        ax.plot([0, 0], [0, 0], [0, 0], '-', lw=2, color=col, alpha=1.,
                    label=config['components'][c]['name'])
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend([plt.Line2D((0, 1), (0, 0), lw=2, color=colors['comps'][(c) %
                                len(colors['comps'])]) for c, handle in enumerate(
                                handles)], labels, loc=2, bbox_to_anchor=[
                                -.025, .94])
    leg.draw_frame(False)
    for text in leg.get_texts():
        text.set_color(colors['text'])
    
    return fig


# Creates an animated plot of magnetization over time output type='xy' for transversal and 'z' for longitudinal
def plotFrameMT(config, signal, frame, output):
    if output['type'] not in ['xy', 'z']:
        raise Exception('output "type" must be 3D, kspace, psd, xy (transversal) or z (longitudinal)')

    # create diagram
    xmin, xmax = 0, config['clock'][-1]
    if output['type'] == 'xy':
        if 'abs' in output and not output['abs']:
            ymin, ymax = -1, 1
        else:
            ymin, ymax = 0, 1
    elif output['type'] == 'z':
        ymin, ymax = -1, 1
    fig = plt.figure(figsize=(5, 2.7), facecolor=colors['bg'])
    ax = fig.gca(xlim=(xmin, xmax), ylim=(ymin, ymax), fc=colors['bg'])
    for side in ['bottom', 'right', 'top', 'left']:
        ax.spines[side].set_visible(False)  # remove default axes
    ax.grid()
    plt.title(config['title'], color=colors['text'])
    plt.xlabel('time[ms]', horizontalalignment='right', color=colors['text'])
    if output['type'] == 'xy':
        if 'abs' in output and not output['abs']:
            ax.xaxis.set_label_coords(1.1, .475)
            plt.ylabel('$M_x, M_y$', rotation=0, color=colors['text'])
        else: # absolute value of transversal magnetization
            ax.xaxis.set_label_coords(1.1, .1)
            plt.ylabel('$|M_{xy}|$', rotation=0, color=colors['text'])
    elif output['type'] == 'z':
        ax.xaxis.set_label_coords(1.1, .475)
        plt.ylabel('$M_z$', rotation=0, color=colors['text'])
    ax.yaxis.set_label_coords(-.07, .95)
    plt.tick_params(axis='y', labelleft='off')
    plt.tick_params(axis='x', colors=colors['text'])
    ax.xaxis.set_ticks_position('none')  # tick markers
    ax.yaxis.set_ticks_position('none')

    # draw x and y axes as arrows
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height  # get width and height of axes object
    hw = 1/25*(ymax-ymin)  # manual arrowhead width and length
    hl = 1/25*(xmax-xmin)
    yhw = hw/(ymax-ymin)*(xmax-xmin) * height/width  # compute matching arrowhead length and width
    yhl = hl/(xmax-xmin)*(ymax-ymin) * width/height
    ax.arrow(xmin, 0, (xmax-xmin)*1.05, 0, fc=colors['text'], ec=colors['text'], lw=1, head_width=hw, head_length=hl, clip_on=False, zorder=100)
    ax.arrow(0, ymin, 0, (ymax-ymin)*1.05, fc=colors['text'], ec=colors['text'], lw=1, head_width=yhw, head_length=yhl, clip_on=False, zorder=100)
    
    # Draw magnetization vectors
    nComps = signal.shape[0]
    if output['type'] == 'xy':
        for c in range(nComps):
            col = colors['comps'][c % len(colors['comps'])]
            if 'abs' in output and not output['abs']: # real and imag part of transversal magnetization
                ax.plot(config['clock'][:frame+1], signal[c,0,:frame+1], '-', lw=2, color=col)
                col = colors['comps'][c+nComps+1 % len(colors['comps'])]
                ax.plot(config['clock'][:frame+1], signal[c,1,:frame+1], '-', lw=2, color=col)
            else: # absolute value of transversal magnetization
                ax.plot(config['clock'][:frame+1], np.linalg.norm(signal[c,:2,:frame+1], axis=0), '-', lw=2, color=col)
        # plot sum component if both water and fat (special case)
        if all(key in [comp['name'] for comp in config['components']] for key in ['water', 'fat']):
            col = colors['comps'][nComps % len(colors['comps'])]
            if 'abs' in output and not output['abs']: # real and imag part of transversal magnetization
                ax.plot(config['clock'][:frame+1], np.mean(signal[:,0,:frame+1],0), '-', lw=2, color=col)
                col = colors['comps'][2*nComps+1 % len(colors['comps'])]
                ax.plot(config['clock'][:frame+1], np.mean(signal[:,1,:frame+1],0), '-', lw=2, color=col)
            else: # absolute value of transversal magnetization
                ax.plot(config['clock'][:frame+1], np.linalg.norm(np.mean(signal[:,:2,:frame+1],0), axis=0), '-', lw=2, color=col)

    elif output['type'] == 'z':
        for c in range(nComps):
            col = colors['comps'][(c) % len(colors['comps'])]
            ax.plot(config['clock'][:frame+1], signal[c,2,:frame+1], '-', lw=2, color=col)

    return fig


def plotFrameKspace(config, frame):
    #TODO: support for 3D k-space
    kmax = 1/(2*config['locSpacing'])
    xmin, xmax = -kmax, kmax
    ymin, ymax = -kmax, kmax
    fig = plt.figure(figsize=(5, 5), facecolor=colors['bg'])
    ax = fig.gca(xlim=(xmin, xmax), ylim=(ymin, ymax), fc=colors['bg'])
    for side in ['bottom', 'right', 'top', 'left']:
        ax.spines[side].set_color(colors['text'])
    ax.grid()
    plt.title(config['title'], color=colors['text'])
    plt.xlabel('$k_x$ [m$^{-1}$]', horizontalalignment='right', color=colors['text'])
    plt.ylabel('$k_y$ [m$^{-1}$]', rotation=0, color=colors['text'])
    plt.tick_params(axis='y', colors=colors['text'])
    plt.tick_params(axis='x', colors=colors['text'])

    kx, ky, kz = 0, 0, 0
    for event in config['pulseSeq']:
        if event['frame']<frame:
            if any([key in event for key in ['Gx', 'Gy', 'Gz']]):
                dur = config['dt']*(min(event['frame']+event['nFrames'], frame)-event['frame'])
                if 'Gx' in event:
                    kx += gyro*event['Gx']*dur/1000
                if 'Gy' in event:
                    ky += gyro*event['Gy']*dur/1000
                if 'Gz' in event:
                    kz += gyro*event['Gz']*dur/1000
            elif 'spoil' in event and event['spoil']:
                kx, ky, kz = 0, 0, 0
        else: 
            break
    ax.plot(kx, ky, '.', markersize=10, color=colors['kSpacePos'])
    return fig


def plotFramePSD(config, frame):
    xmin, xmax = 0, config['kernelClock'][-1]
    ymin, ymax = 0, 5
    fig = plt.figure(figsize=(5, 5), facecolor=colors['bg'])
    ax = fig.gca(xlim=(xmin, xmax), ylim=(ymin, ymax), fc=colors['bg'])
    for side in ['bottom', 'right', 'top', 'left']:
        ax.spines[side].set_visible(False)  # remove default axes
    plt.title(config['title'], color=colors['text'])
    plt.xlabel('time[ms]', horizontalalignment='right', color=colors['text'])
    #ax.xaxis.set_label_coords(1.1, .1)
    #plt.ylabel('$|M_{xy}|$', rotation=0, color=colors['text'])
    plt.tick_params(axis='y', labelleft='off')
    plt.tick_params(axis='x', colors=colors['text'])
    ax.xaxis.set_ticks_position('none')  # tick markers
    ax.yaxis.set_ticks_position('none')

    # draw x and y axes as arrows
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height  # get width and height of axes object
    hw = 1/25*(ymax-ymin)  # manual arrowhead width and length
    hl = 1/25*(xmax-xmin)
    yhw = hw/(ymax-ymin)*(xmax-xmin) * height/width  # compute matching arrowhead length and width
    yhl = hl/(xmax-xmin)*(ymax-ymin) * width/height
    ax.arrow(xmin, 0, (xmax-xmin)*1.05, 0, fc=colors['text'], ec=colors['text'], lw=1, head_width=hw, head_length=hl, clip_on=False, zorder=100)
    
    ylim = {}
    B1s = [event['B1'] for event in config['pulseSeq'] if 'B1' in event and event['B1']!='inf']
    if len(B1s)>0:
        ylim['B1'] = 2.1*np.max(B1s)
    Gxs = [np.abs(event['Gx']) for event in config['pulseSeq'] if 'Gx' in event]
    Gys = [np.abs(event['Gy']) for event in config['pulseSeq'] if 'Gy' in event]
    Gzs = [np.abs(event['Gz']) for event in config['pulseSeq'] if 'Gz' in event]
    ylim['Gx'] = ylim['Gy'] = ylim['Gz'] = 2.1*np.max(np.concatenate((Gxs, Gys, Gzs)))
    ypos = {board: y for board, y in [('B1',4), ('Gx',3), ('Gy',2), ('Gz',1)]}
    for event in config['pulseSeq']:
        xpos = config['kernelClock'][event['frame']]
        w = config['kernelClock'][event['frame']+event['nFrames']] - xpos
        for board in ['B1', 'Gx', 'Gy', 'Gz']:
            if board in event:
                if event[board]=='inf':
                    h = 1/2.1
                else:
                    h = .9*event[board]/ylim[board]
                ax.add_patch(Rectangle((xpos, ypos[board]), w, h, lw=1, facecolor=colors['boards'][board], edgecolor=colors['text']))
    for board in ['B1', 'Gx', 'Gy', 'Gz']:
        ax.plot([xmin, xmax], [ypos[board], ypos[board]], color=colors['text'], lw=1, clip_on=False, zorder=100)
        ax.text(0, ypos[board], board, fontsize=14,
            color=colors['text'], horizontalalignment='right', verticalalignment='center')
    ax.plot([config['kernelClock'][frame%(len(config['kernelClock'])-1)], config['kernelClock'][frame%(len(config['kernelClock'])-1)]], [0, 5], color=colors['text'], lw=1, clip_on=False, zorder=100)
    return fig


def radians(degrees): return degrees*np.pi/180


def degrees(radians): return radians*180./np.pi


# Apply spoiling of the transversal magnetization
def spoil(M): return np.array([0, 0, M[2]])


def derivs(M, t, Meq, w, w1, T1, T2):  # Bloch equations in rotating frame
    dMdt = np.zeros_like(M)
    dMdt[0] = -M[0]/T2+M[1]*w+M[2]*w1.real
    dMdt[1] = -M[0]*w-M[1]/T2+M[2]*w1.imag
    dMdt[2] = -M[0]*w1.real-M[1]*w1.imag+(Meq-M[2])/T1
    return dMdt


# Simulate magnetization vector during nTR applications of pulseSeq
def applyPulseSeq(config, Meq, w, T1, T2, xpos=0, ypos=0, zpos=0):
    M = np.zeros([config['nFrames']+1, 3])
    M[0] = [0, 0, Meq] # Initial state is equilibrium magnetization
    for rep in range(config['nTR']):
        TRframe = rep * config['nFramesPerTR'] #starting frame of TR
        frame = 0 #frame within TR
        for event in config['pulseSeq']:
            # Relaxation up to event
            T = config['kernelClock'][event['frame']]-config['kernelClock'][frame]
            t = np.linspace(0, T, event['frame']-frame+1, endpoint=True)
            M[TRframe+frame:TRframe+event['frame']+1] = integrate.odeint(derivs, M[TRframe+frame], t, args=(Meq, w, 0., T1, T2))
            frame = event['frame']

            wg = w  # frequency due to w plus any gradients
            if 'w1' in event:
                w1 = event['w1']
            else:
                w1 = 0
            t = np.linspace(0, event['nFrames']*config['dt'], event['nFrames']+1, endpoint=True)

            if 'spoil' in event and event['spoil']: # Spoiler event
                M[TRframe+frame] = spoil(M[TRframe+frame])
            else:
                if 'FA' in event and event['dur']==0: # "instant" RF-pulse event (incompatible with gradient)
                    M[TRframe+frame:TRframe+frame+event['nFrames']+1] = integrate.odeint(derivs, M[TRframe+frame], t, args=(Meq, 0., w1, np.inf, np.inf))
                else: # RF-pulse and/or gradient event
                    if any(key in event for key in ['Gx', 'Gy', 'Gz']): # Gradient present
                        if 'Gx' in event:
                            wg += 2*np.pi*gyro*event['Gx']*xpos/1000 # [krad/s]
                        if 'Gy' in event:
                            wg += 2*np.pi*gyro*event['Gy']*ypos/1000 # [krad/s]
                        if 'Gz' in event:
                            wg += 2*np.pi*gyro*event['Gz']*zpos/1000 # [krad/s]
                    M[TRframe+frame:TRframe+frame+event['nFrames']+1] = integrate.odeint(derivs, M[TRframe+frame], t, args=(Meq, wg, w1, T1, T2))

            frame += event['nFrames']

        # Then relaxation until end of TR
        T = config['kernelClock'][-1]-config['kernelClock'][frame]
        t = np.linspace(0, T, config['nFramesPerTR']-frame+1, endpoint=True)
        M1 = integrate.odeint(derivs, M[TRframe+frame], t, args=(Meq, w, 0., T1, T2))
        M[TRframe+frame:(rep+1)*config['nFramesPerTR']+1] = M1
    return M[:-1].transpose()


# Simulate Nisochromats dephasing magnetization vectors per component
def simulateComponent(config, component, Meq, xpos=0, ypos=0, zpos=0):
    # Shifts in ppm for dephasing vectors:
    isochromats = [(2*i+1-config['nIsochromats'])/2*config['isochromatStep']+component['CS'] for i in range(0, config['nIsochromats'])]
    comp = np.empty((config['nIsochromats'],3,config['nFrames']))

    for m, isochromat in enumerate(isochromats):
        w = config['w0']*isochromat*1e-6  # Demodulated frequency [krad]
        comp[m,:,:] = applyPulseSeq(config, Meq, w, component['T1'], component['T2'], xpos, ypos, zpos)
    return comp


# Get opacity and text for spoiler and RF text flashes in 3D plot
def getText(config):
    framesSinceSpoil = np.full([config['nFrames']+1], np.inf, dtype=int)
    framesSinceRF = np.full([config['nFrames']+1], np.inf, dtype=int)
    framesSinceG = np.full([config['nFrames']+1], np.inf, dtype=int)
    config['RFtext'] = np.full([config['nFrames']+1], '', dtype=object)
    config['Gtext'] = np.full([config['nFrames']+1], '', dtype=object)
    
    for rep in range(config['nTR']):
        TRframe = rep * config['nFramesPerTR'] #starting frame of TR
        frame = 0 #frame within TR
        for event in config['pulseSeq']:
            # Frames during relaxation
            nFrames = event['frame']-frame
            count = np.linspace(0, nFrames, nFrames+1, endpoint=True)
            framesSinceSpoil[TRframe+frame:TRframe+event['frame']+1] = framesSinceSpoil[TRframe+frame] + count
            framesSinceRF[TRframe+frame:TRframe+event['frame']+1] = framesSinceRF[TRframe+frame] + count
            framesSinceG[TRframe+frame:TRframe+event['frame']+1] = framesSinceG[TRframe+frame] + count
            frame = event['frame']

            #Frames during event
            count = np.linspace(0, event['nFrames'], event['nFrames']+1, endpoint=True)
            
            if 'spoil' in event and event['spoil']: # Spoiler event
                framesSinceSpoil[TRframe+frame] = 0
            framesSinceSpoil[TRframe+frame:TRframe+frame+event['nFrames']+1] = framesSinceSpoil[TRframe+frame] + count
            
            if 'FA' in event: # RF-pulse and/or gradient event
                framesSinceRF[TRframe+frame:TRframe+frame+event['nFrames']+1] = 0
                # TODO: add info about the RF phase angle
                config['RFtext'][TRframe+frame:] = str(int(abs(event['FA'])))+u'\N{DEGREE SIGN}'+'-pulse'
            else:
                framesSinceRF[TRframe+frame:TRframe+frame+event['nFrames']+1] = framesSinceRF[TRframe+frame] + count
            if any(key in event for key in ['Gx', 'Gy', 'Gz']): # gradient event
                framesSinceG[TRframe+frame:TRframe+frame+event['nFrames']+1] = 0
                grad = ''
                for g in ['Gx', 'Gy', 'Gz']:
                    if g in event:
                        grad += '  {}: {} mT/m'.format(g, event[g])
                config['Gtext'][TRframe+frame:] = grad
            else:
                framesSinceG[TRframe+frame:TRframe+frame+event['nFrames']+1] = framesSinceG[TRframe+frame] + count
            
            frame += event['nFrames']

        # Frames during relaxation until end of TR
        nFrames = config['nFramesPerTR']-frame
        count = np.linspace(0, nFrames, nFrames+1, endpoint=True)
        framesSinceSpoil[TRframe+frame:(rep+1)*config['nFramesPerTR']+1] = framesSinceSpoil[TRframe+frame] + count
        framesSinceRF[TRframe+frame:(rep+1)*config['nFramesPerTR']+1] = framesSinceRF[TRframe+frame] + count
        framesSinceG[TRframe+frame:(rep+1)*config['nFramesPerTR']+1] = framesSinceG[TRframe+frame] + count
            
    # Calculate alphas (one second fade)
    config['spoilAlpha'] = [max(1.0-frames/config['fps'], 0) for frames in framesSinceSpoil]
    config['Galpha'] = [max(1.0-frames/config['fps'], 0) for frames in framesSinceG]
    config['RFalpha'] = [max(1.0-frames/config['fps'], 0) for frames in framesSinceRF]


def checkPulseSeq(config):
    if 'pulseSeq' not in config:
        config['pulseSeq'] = []
    allowedKeys = ['t', 'spoil', 'dur', 'FA', 'B1', 'phase', 'Gx', 'Gy', 'Gz']
    for event in config['pulseSeq']:
        for item in event.keys(): # allowed keys
            if item not in allowedKeys:
                raise Exception('PulseSeq key "{}" not supported'.format(item))
        if not 't' in event:
            raise Exception('All pulseSeq events must have an event time "t"')    

    # Sort pulseSeq according to event time
    config['pulseSeq'] = sorted(config['pulseSeq'], key=lambda event: event['t'])
    
    t = np.array([0.0])
    dt = config['dt']
    for event in config['pulseSeq']:
        T = np.round(event['t']/dt)*dt-t[-1] # time up to event
        if T<0:
            raise Exception('Pulse sequence events overlap')
        t = np.append(t[:-1], t[-1]+np.linspace(0, T, np.round(T/dt)+1, endpoint=True))
        event['frame'] = len(t)-1 # starting frame of event
            
        if 'spoil' in event: # Spoiler event
            if any(key in event for key in ['dur', 'FA', 'B1', 'phase', 'Gx', 'Gy', 'Gz']):
                raise Exception('Spoiler event should only have event time t and spoil: true')
            event['dur'] = 0
            event['nFrames'] = 0

        if 'FA' in event or 'B1' in event: # RF-pulse (possibly with gradient)
            if all(key in event for key in ['FA', 'B1', 'dur']):
                raise Exception('RF-pulse over-determined. Provide only two out of "FA", "B1", and "dur"')
            if 'B1' not in event and 'dur' not in event:
                if 'B1' in config:
                    event['B1'] = config['B1'] # use "global" B1
                else:
                    event['B1'] = 'inf' # "instant" RF-pulse            
            if 'B1' in event: 
                if event['B1'] == 'inf': # handle "instant" RF pulse, specified by B1: inf
                    if 'dur' in event:
                        raise Exception('Cannot combine given dur with "infinite" B1 pulse')
                    event['dur'] = 0
            else:
                if event['dur']==0:
                    event['B1'] = 'inf'
                else:
                    event['B1'] = abs(event['FA'])/(event['dur']*360*gyro*1e-6)
            if 'dur' not in event:
                event['dur'] = abs(event['FA'])/(360*gyro*event['B1']*1e-6) # RF pulse duration
            if 'FA' not in event:
                event['FA'] = 360*(event['dur']*gyro*event['B1']*1e-6) # calculate prescribed FA
            if event['dur']>0:
                event['nFrames'] = int(max(np.round(event['dur']/dt), 1))
            else:
                event['nFrames'] = int(np.round(abs(event['FA'])*config['fps']/90)) # one sec per 90 flip
            if 'phase' in event: # Set complex flip angles
                event['FA'] = event['FA']*np.exp(1j*radians(event['phase']))
            event['w1'] = radians(event['FA'])/(event['nFrames']*dt)

        if any(key in event for key in ['Gx', 'Gy', 'Gz']): # Gradient (no RF)
            if not ('dur' in event and event['dur']>0):
                raise Exception('Gradient must have a specified duration>0 (dur [ms])')
            if 'FA' not in event and 'phase' in event:
                raise Exception('Gradient event should have no phase')
            event['nFrames'] = int(max(np.round(event['dur']/dt), 1))
            if 'Gx' in event:
                Gx = event['Gx']*event['nFrames']*dt/event['dur'] # adjust Gx for truncation of duration
            if 'Gy' in event:
                Gy = event['Gy']*event['nFrames']*dt/event['dur'] # adjust Gy for truncation of duration
            if 'Gz' in event:
                Gz = event['Gz']*event['nFrames']*dt/event['dur'] # adjust Gz for truncation of duration
            
        if event['dur']>0:
            event['dur'] = event['nFrames']*dt  # truncate duration to whole frames
        t = np.append(t[:-1], t[-1]+np.linspace(0, event['dur'], event['nFrames']+1, endpoint=True))
        
    T = np.ceil(config['TR']/dt)*dt-t[-1] # time up to TR
    if np.round(T/dt)<0:
        raise Exception('Pulse sequence events not within TR')
    t = np.append(t[:-1], t[-1]+np.linspace(0, T, np.round(T/dt)+1, endpoint=True))
    config['kernelClock'] = t
    config['clock'] = np.array([0])
    for rep in range(config['nTR']):
        config['clock'] = np.append(config['clock'][:-1], config['clock'][-1]+t)

def arrangeLocations(slices, config):
    if not isinstance(slices, list):
        raise Exception('Did not expect {} in config "locations"'.format(type(slices)))
    if not isinstance(slices[0], list):
        slices = [slices]
    if not isinstance(slices[0][0], list):
        slices = [slices]
    if 'nz' not in config:
        config['nz'] = len(slices)
    elif len(slices)!=config['nz']:
        raise Exception('Config "locations": number of slices do not match')
    if 'ny' not in config:
        config['ny'] = len(slices[0])
    elif len(slices[0])!=config['ny']:
        raise Exception('Config "locations": number of rows do not match')
    if 'nx' not in config:
        config['nx'] = len(slices[0][0])
    elif  len(slices[0][0])!=config['nx']:
        raise Exception('Config "locations": number of elements do not match')
    return slices



def checkConfig(config):
    if any([key not in config for key in ['TR', 'B0', 'speed', 'output']]):
        raise Exception('Config must contain "TR", "B0", "speed", and "output"')
    if 'title' not in config:
        config['title'] = ''
    if 'nTR' not in config:
        config['nTR'] = 1
    if 'nIsochromats' not in config:
        config['nIsochromats'] = 1
    if 'isochromatStep' not in config:
        if config['nIsochromats']>1:
            raise Exception('Please specify "isochromatStep" [ppm] in config')
        else:
            config['isochromatStep']=0
    if 'components' not in config:
        config['components'] = [{}]
    for comp in config['components']:
        for (key, default) in [('name', ''), ('CS', 0), ('T1', np.inf), ('T2', np.inf)]:
            if key not in comp:
                comp[key] = default
    config['nComps'] = len(config['components'])
    if 'locSpacing' not in config:
        config['locSpacing'] = 0.001      # distance between locations [m]

    if 'fps' not in config:
        config['fps'] = 15 # Frames per second in animation (<=15 should be supported by powepoint)
    
    # calculations
    config['dt'] = 1e3/config['fps']*config['speed'] # Time resolution [msec]
    config['w0'] = 2*np.pi*gyro*config['B0'] # Larmor frequency [kRad/s]

    checkPulseSeq(config)
    config['nFrames'] = len(config['clock'])-1
    config['nFramesPerTR'] = len(config['kernelClock'])-1

    ### Arrange locations ###
    if not 'locations' in config:
        config['locations'] = arrangeLocations([[[1]]], config)
    else:
        nx, ny, nz = [], [], []
        if isinstance(config['locations'], dict):
            for comp in iter(config['locations']):
                config['locations'][comp] = arrangeLocations(config['locations'][comp], config)
        else:
            config['locations'] = arrangeLocations(config['locations'], config)
    for (FOV, n) in [('FOVx', 'nx'), ('FOVy', 'ny'), ('FOVz', 'nz')]:
        if FOV not in config:
            config[FOV] = config[n]*config['locSpacing'] #FOV in m


# Main program
def BlochBuster(configFile, leapFactor=1, blackBackground=False, useFFMPEG = True):
    # Set global constants
    global gyro
    gyro = 42577.			# Gyromagnetic ratio [kHz/T]

    if blackBackground:
        for i in ['bg', 'axis', 'text', 'circle']:
            colors[i][:3] = list(map(lambda x: 1-x, colors[i][:3]))
    # Read configuration file
    with open(configFile, 'r') as f:
        try:
            config = yaml.load(f)
        except yaml.YAMLError as exc:
            raise Exception('Error reading config file') from exc
    
    ### Setup config correctly ###
    checkConfig(config)

    ### Simulate ###
    vectors = np.empty((config['nx'],config['ny'],config['nz'],config['nComps'],config['nIsochromats'],3,config['nFrames']))
    for z in range(config['nz']):
        for y in range(config['ny']):
            for x in range(config['nx']):
                for c, component in enumerate(config['components']):
                    if component['name'] in config['locations']:
                        try:
                            Meq = config['locations'][component['name']][z][y][x]
                        except:
                            raise Exception('Is the location matrix shape equal for all components?')
                    elif isinstance(config['locations'], list):
                        Meq = config['locations'][z][y][x]
                    else:
                        Meq = 0.0
                    xpos = (x+.5-config['nx']/2)*config['locSpacing']
                    ypos = (y+.5-config['ny']/2)*config['locSpacing']
                    zpos = (z+.5-config['nz']/2)*config['locSpacing']
                    vectors[x,y,z,c,:,:,:] = simulateComponent(config, component, Meq, xpos, ypos, zpos)
    signal = np.mean(vectors, (0,1,2,4)) # sum over space and isochromats

    ### Animate ###
    getText(config) # prepare text flashes for 3D plot 
    delay = int(100/config['fps']*leapFactor)  # Delay between frames in ticks of 1/100 sec

    tmpdir = './tmp'
    outdir = './out'
    if os.path.isdir(tmpdir):
        rmTmpDir = input('Temporary folder "{}" already exists. Delete(Y/N)?'.format(tmpdir))
        if rmTmpDir.upper() == 'Y':
            shutil.rmtree(tmpdir)
        else:
            raise Exception('No files written.')
    for output in config['output']:
        if output['file']:
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, output['file'])
            if useFFMPEG:
                ffmpegWriter = FFMPEGwriter.FFMPEGwriter(config['fps'])
            else:
                os.makedirs(tmpdir, exist_ok=True)
            for frame in range(0, config['nFrames'], leapFactor):
                # Use only every leapFactor frame in animation
                if output['type'] == '3D':
                    fig = plotFrame3D(config, vectors, frame)
                elif output['type'] == 'kspace':
                    fig = plotFrameKspace(config, frame)
                elif output['type'] == 'psd':
                    fig = plotFramePSD(config, frame)
                else:
                    fig = plotFrameMT(config, signal, frame, output)
                plt.draw()
                if useFFMPEG:
                    ffmpegWriter.addFrame(fig)
                else: # use imagemagick: save frames temporarily 
                    file = os.path.join(tmpdir, '{}.png'.format(str(frame).zfill(4)))
                    print('Saving frame {}/{} as "{}"'.format(frame+1, config['nFrames'], file))
                    plt.savefig(file, facecolor=plt.gcf().get_facecolor())
                plt.close()
            if useFFMPEG:
                ffmpegWriter.write(outfile)
            else: # use imagemagick
                print('Creating animated gif "{}"'.format(outfile))
                compress = '-layers Optimize'
                os.system(('convert {} -delay {} {}/*png {}'.format(compress, delay, tmpdir, outfile)))
                shutil.rmtree(tmpdir)


# Command line parser
def main():
    # Initiate command line parser
    p = optparse.OptionParser()
    p.add_option('--configFile', '-c', default='',  type="string", help="Name of configuration text file")
    p.add_option('--leapFactor', '-l', default=1, type="int", help="Leap factor for smaller filesize and fewer frames per second")
    p.add_option('--blackBackground', '-b', action="store_true", dest="blackBackground", default=False, help="Plot with black background")
    # Parse command line
    options, arguments = p.parse_args()
    BlochBuster(options.configFile, options.leapFactor, options.blackBackground)

if __name__ == '__main__':
    main()
