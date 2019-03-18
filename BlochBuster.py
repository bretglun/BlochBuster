#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Johan Berglund
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
'''
BlochBuster is a nuclear magnetic resonance Bloch equation simulator written in Python. 
It simulates magnetization vectors based on the Bloch equations, including precession, relaxation, and excitation. 
BlochBuster outputs animated gif or mp4 files, which can be 3D plots of the magnetization vectors, plots of transverse and longitudinal magnetization, or pulse sequence diagrams.
Input paramaters are provided by human readable configuration files.
The animations are made using ffmpeg.
'''

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
import argparse
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
            'boards': { 'w1': [.5,0,0],
                        'Gx': [0,.5,0],
                        'Gy': [0,.5,0],
                        'Gz': [0,.5,0]
                        },
            'kSpacePos': [1, .5, 0]
            }

class Arrow3D(FancyArrowPatch):
    '''Matplotlib FancyArrowPatch for 3D rendering.'''
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def plotFrame3D(config, vectors, frame, output):
    '''Creates a plot of magnetization vectors in a 3D view.
    
    Args:
        config: configuration dictionary.
	vectors:    numpy array of size [nx, ny, nz, nComps, nIsochromats, 3, nFrames].
        frame:  which frame to plot.
        output: specification of desired output (dictionary from config).

    Returns:
        plot figure.

    '''
    nx, ny, nz, nComps, nIsoc = vectors.shape[:5]
    if config['collapseLocations']:
        xpos = np.zeros([nx])
        ypos = np.zeros([nx])
        zpos = np.zeros([nx])
    else:
        xpos = np.arange(nx)-nx/2+.5
        ypos = -(np.arange(ny)-ny/2+.5)
        zpos = -(np.arange(nz)-nz/2+.5)

    # Create 3D axes
    if nx*ny*nz==1 or config['collapseLocations']:
        aspect = .952 # figure aspect ratio
    elif nz==1 and ny==1 and nx>1:
        aspect = 0.6
    elif nz==1 and nx>1 and ny>1:
        aspect = .75
    else:
        aspect = 1
    figSize = 5 # figure size in inches
    canvasWidth = figSize
    canvasHeight = figSize*aspect
    fig = plt.figure(figsize=(canvasWidth, canvasHeight), dpi=output['dpi'])
    axLimit = max(nx,ny,nz)/2+.5
    if config['collapseLocations']:
        axLimit = 1.0
    ax = fig.gca(projection='3d', xlim=(-axLimit,axLimit), ylim=(-axLimit,axLimit), zlim=(-axLimit,axLimit), fc=colors['bg'])
    ax.set_aspect('equal')
    
    if nx*ny*nz>1 and not config['collapseLocations']:
        azim = -78 # azimuthal angle of x-y-plane
        ax.view_init(azim=azim) #ax.view_init(azim=azim, elev=elev)
    ax.set_axis_off()
    width = 1.65 # to get tight cropping
    height = width/aspect
    left = (1-width)/2
    bottom = (1-height)/2
    if nx*ny*nz==1 or config['collapseLocations']: # shift to fit legend
        left += .035
        bottom += -.075
    else:
        bottom += -.085
    ax.set_position([left, bottom, width, height])

    if output['drawAxes']:
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
    time_text = fig.text(0, 0, 'time = %.1f msec' % (config['tFrames'][frame%(len(config['t'])-1)]), color=colors['text'], verticalalignment='bottom')

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
                        thres = 0.075*axLimit
                        if Mnorm>thres:
                            arrowScale = 20
                        else:
                            arrowScale = 20*Mnorm/thres # Shrink arrowhead close to origo
                        ax.add_artist(Arrow3D(  [xpos[x], xpos[x]+M[0]], 
                                                [ypos[y], ypos[y]+M[1]],
                                                [zpos[z], zpos[z]+M[2]], 
                                                mutation_scale=arrowScale,
                                                arrowstyle='-|>', shrinkA=0, shrinkB=0, lw=2,
                                                color=col, alpha=alpha, 
                                                zorder=order[m]+nIsoc*int(100*(1-Mnorm))))

    # Draw "spoiler" and "FA-pulse" text
    fig.text(1, .94, config['RFtext'][frame], fontsize=14, alpha=config['RFalpha'][frame],
            color=colors['RFtext'], horizontalalignment='right', verticalalignment='top')
    fig.text(1, .88, config['Gtext'][frame], fontsize=14, alpha=config['Galpha'][frame],
            color=colors['Gtext'], horizontalalignment='right', verticalalignment='top')
    fig.text(1, .82, config['spoiltext'], fontsize=14, alpha=config['spoilAlpha'][frame],
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


def plotFrameMT(config, signal, frame, output):
    '''Creates a plot of transversal or longituinal magnetization over time.
    
    Args:
        config: configuration dictionary.
	signal: numpy array of size [nComps, 3, nFrames].
        frame:  which frame to plot up to.
        output: specification of desired output (dictionary from config).

    Returns:
        plot figure.

    '''
    if output['type'] not in ['xy', 'z']:
        raise Exception('output "type" must be 3D, kspace, psd, xy (transversal) or z (longitudinal)')

    # create diagram
    xmin, xmax = 0, config['t'][-1]
    if output['type'] == 'xy':
        if 'abs' in output and not output['abs']:
            ymin, ymax = -1, 1
        else:
            ymin, ymax = 0, 1
    elif output['type'] == 'z':
        ymin, ymax = -1, 1
    fig = plt.figure(figsize=(5, 2.7), facecolor=colors['bg'], dpi=output['dpi'])
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
                ax.plot(config['t'][:frame+1], signal[c,0,:frame+1], '-', lw=2, color=col)
                col = colors['comps'][c+nComps+1 % len(colors['comps'])]
                ax.plot(config['t'][:frame+1], signal[c,1,:frame+1], '-', lw=2, color=col)
            else: # absolute value of transversal magnetization
                ax.plot(config['t'][:frame+1], np.linalg.norm(signal[c,:2,:frame+1], axis=0), '-', lw=2, color=col)
        # plot sum component if both water and fat (special case)
        if all(key in [comp['name'] for comp in config['components']] for key in ['water', 'fat']):
            col = colors['comps'][nComps % len(colors['comps'])]
            if 'abs' in output and not output['abs']: # real and imag part of transversal magnetization
                ax.plot(config['t'][:frame+1], np.mean(signal[:,0,:frame+1],0), '-', lw=2, color=col)
                col = colors['comps'][2*nComps+1 % len(colors['comps'])]
                ax.plot(config['t'][:frame+1], np.mean(signal[:,1,:frame+1],0), '-', lw=2, color=col)
            else: # absolute value of transversal magnetization
                ax.plot(config['t'][:frame+1], np.linalg.norm(np.mean(signal[:,:2,:frame+1],0), axis=0), '-', lw=2, color=col)

    elif output['type'] == 'z':
        for c in range(nComps):
            col = colors['comps'][(c) % len(colors['comps'])]
            ax.plot(config['t'][:frame+1], signal[c,2,:frame+1], '-', lw=2, color=col)

    return fig


def plotFrameKspace(config, frame, output):
    '''Creates a plot of k-space position for the given frame.
    
    Args:
        config: configuration dictionary.
        frame:  which frame to plot.
        output: specification of desired output (dictionary from config).

    Returns:
        plot figure.

    '''
    #TODO: support for 3D k-space
    kmax = 1/(2*config['locSpacing'])
    xmin, xmax = -kmax, kmax
    ymin, ymax = -kmax, kmax
    fig = plt.figure(figsize=(5, 5), facecolor=colors['bg'], dpi=output['dpi'])
    ax = fig.gca(xlim=(xmin, xmax), ylim=(ymin, ymax), fc=colors['bg'])
    for side in ['bottom', 'right', 'top', 'left']:
        ax.spines[side].set_color(colors['text'])
    ax.grid()
    plt.title(config['title'], color=colors['text'])
    plt.xlabel('$k_x$ [m$^{-1}$]', horizontalalignment='right', color=colors['text'])
    plt.ylabel('$k_y$ [m$^{-1}$]', rotation=0, color=colors['text'])
    plt.tick_params(axis='y', colors=colors['text'])
    plt.tick_params(axis='x', colors=colors['text'])

    frameTime = config['tFrames'][frame]%config['TR']
    kx, ky, kz = 0, 0, 0
    for i, event in enumerate(config['events']):
        firstFrame, lastFrame = getEventFrames(config, i)
        if event['t'] < frameTime:
            dur = min(frameTime, config['t'][lastFrame])-config['t'][firstFrame]
            if 'spoil' in event and event['spoil']:
                kx, ky, kz = 0, 0, 0
            kx += gyro * event['Gx'] * dur / 1e3
            ky += gyro * event['Gy'] * dur / 1e3
            kz += gyro * event['Gz'] * dur / 1e3
        else:
            break
    ax.plot(kx, ky, '.', markersize=10, color=colors['kSpacePos'])
    return fig


def plotFramePSD(config, frame, output):
    '''Creates a plot of the pulse sequence diagram.
    
    Args:
        config: configuration dictionary.
        frame:  which frame to indicate by vertical line.
        output: specification of desired output (dictionary from config).

    Returns:
        plot figure.

    '''
    if 'fig' in output:
        fig, timeLine = output['fig']
    else:
        xmin, xmax = 0, config['kernelClock'][-1]
        ymin, ymax = 0, 5
        fig = plt.figure(figsize=(5, 5), facecolor=colors['bg'], dpi=output['dpi'])
        ax = fig.gca(xlim=(xmin, xmax), ylim=(ymin, ymax), fc=colors['bg'])
        for side in ['bottom', 'right', 'top', 'left']:
            ax.spines[side].set_visible(False)  # remove default axes
        plt.title(config['title'], color=colors['text'])
        plt.xlabel('time[ms]', horizontalalignment='right', color=colors['text'])
        plt.tick_params(axis='y', labelleft='off')
        plt.tick_params(axis='x', colors=colors['text'])
        ax.xaxis.set_ticks_position('none')  # tick markers
        ax.yaxis.set_ticks_position('none')

        # draw x axis as arrow
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height  # get width and height of axes object
        hw = 1/25*(ymax-ymin)  # manual arrowhead width and length
        hl = 1/25*(xmax-xmin)
        yhw = hw/(ymax-ymin)*(xmax-xmin) * height/width  # compute matching arrowhead length and width
        yhl = hl/(xmax-xmin)*(ymax-ymin) * width/height
        ax.arrow(xmin, 0, (xmax-xmin)*1.05, 0, fc=colors['text'], ec=colors['text'], lw=1, head_width=hw, head_length=hl, clip_on=False, zorder=100)
        
        ylim = {}
        w1s = [event['w1'] for event in config['events']]
        Gxs = [event['Gx'] for event in config['events']]
        Gys = [event['Gy'] for event in config['events']]
        Gzs = [event['Gz'] for event in config['events']]
        ylim['w1'] = 2.1*np.max([np.abs(w1) for w1 in w1s if np.abs(w1) < 50])
        ylim['Gx'] = ylim['Gy'] = ylim['Gz'] = 2.1*np.max(np.concatenate((np.abs(Gxs), np.abs(Gys), np.abs(Gzs))))
        ypos = {board: y for board, y in [('w1',4), ('Gx',3), ('Gy',2), ('Gz',1)]}
        for i, event in enumerate(config['events']):
            firstFrame, lastFrame = getEventFrames(config, i)
            xpos = config['kernelClock'][firstFrame]
            w = config['kernelClock'][lastFrame] - xpos
            for board in ['w1', 'Gx', 'Gy', 'Gz']:
                if board in event:
                    h = .9*event[board]/ylim[board]
                    ax.add_patch(Rectangle((xpos, ypos[board]), w, h, lw=1, facecolor=colors['boards'][board], edgecolor=colors['text']))
        for board in ['w1', 'Gx', 'Gy', 'Gz']:
            ax.plot([xmin, xmax], [ypos[board], ypos[board]], color=colors['text'], lw=1, clip_on=False, zorder=100)
            ax.text(0, ypos[board], board, fontsize=14,
                color=colors['text'], horizontalalignment='right', verticalalignment='center')
    
        # plot vertical time line:
        timeLine, = ax.plot([config['tFrames'][frame]%config['TR'], config['tFrames'][frame]%config['TR']], [0, 5], color=colors['text'], lw=1, clip_on=False, zorder=100)
        output['fig'] = fig, timeLine
    timeLine.set_xdata([config['tFrames'][frame]%config['TR'], config['tFrames'][frame]%config['TR']])
    fig.canvas.draw()
    return fig


# Apply spoiling of the transversal magnetization
def spoil(M):
    '''Spoil the transversal magnetization in magnetization vector.
    
    Args:
        M: magnetization vector, numpy array of size 3.
        
    Returns:
        spoiled magnetization vector, numpy array of size 3.

    '''
    return np.array([0, 0, M[2]])


def derivs(M, t, Meq, w, w1, T1, T2):
    '''Bloch equations in rotating frame.

    Args:
        w:    Larmor frequency :math:`2\\pi\\gamma B_0` [kRad / s].
	w1 (complex):   B1 rotation frequency :math:`2\\pi\\gamma B_1`  [kRad / s].
        T1:   longitudinal relaxation time.
        T2:   transverse relaxation time.
        M:    magnetization vector.
        Meq:  equilibrium magnetization.
        t:    time vector (needed for scipy.integrate.odeint).

    Returns:
        integrand :math:`\\frac{dM}{dt}`
    
    '''
    
    dMdt = np.zeros_like(M)
    dMdt[0] = -M[0]/T2+M[1]*w+M[2]*w1.real
    dMdt[1] = -M[0]*w-M[1]/T2+M[2]*w1.imag
    dMdt[2] = -M[0]*w1.real-M[1]*w1.imag+(Meq-M[2])/T1
    return dMdt


def getEventFrames(config, i):
    '''Get first and last frame of event i in config['events'] in terms of config['t']

    Args:
        config:         configuration dictionary.
        i:              event index
        
    Returns:
        firstFrame:     index of first frame in terms of config['t']
        lastFrame:      index of last frame in terms of config['t']
        
    '''
    try:
        firstFrame = np.where(config['t']==config['events'][i]['t'])[0][0]
    except IndexError:
        print('Event time not found in time vector')
        raise
    
    if i < len(config['events'])-1:
        nextEventTime = config['events'][i+1]['t']
    else:
        nextEventTime = config['TR']
    try:
        lastFrame = np.where(config['t']==nextEventTime)[0][0]
    except IndexError:
        print('Event time not found in time vector')
        raise
    return firstFrame, lastFrame


def applyPulseSeq(config, Meq, M0, w, T1, T2, xpos=0, ypos=0, zpos=0):
    '''Simulate magnetization vector during nTR applications of pulse sequence.
    
    Args:
        config: configuration dictionary.
        Meq:    equilibrium magnetization along z axis.
        M0:     initial state of magnetization vector, numpy array of size 3.
        w:      Larmor frequency :math:`2\\pi\\gamma B_0` [kRad / s].
        T1:     longitudinal relaxation time.
        T2:     transverse relaxation time.
        xpos:   position of magnetization vector along x gradient.
        ypos:   position of magnetization vector along y gradient.
        zpos:   position of magnetization vector along z gradient.
        
    Returns:
        magnetization vector over time, numpy array of size [3, nFrames]

    '''
    M = np.zeros([len(config['t']), 3])
    M[0] = M0 # Initial state

    for rep in range(config['nTR']):
        TRstartFrame = rep * config['nFramesPerTR']

        for i, event in enumerate(config['events']):
            firstFrame, lastFrame = getEventFrames(config, i)
            firstFrame += TRstartFrame
            lastFrame += TRstartFrame

            M0 = M[firstFrame]

            if 'spoil' in event and event['spoil']: # Spoiler event
                M0 = spoil(M0)

            wg = w  # frequency due to w plus any gradients
            wg += 2*np.pi*gyro*event['Gx']*xpos/1000 # [kRad/s]
            wg += 2*np.pi*gyro*event['Gy']*ypos/1000 # [kRad/s]
            wg += 2*np.pi*gyro*event['Gz']*zpos/1000 # [kRad/s]

            w1 = event['w1']

            t = config['t'][firstFrame:lastFrame+1]
            M[firstFrame:lastFrame+1] = integrate.odeint(derivs, M0, t, args=(Meq, wg, w1, T1, T2)) # Solve Bloch equation

    return M.transpose()


def simulateComponent(config, component, Meq, M0=None, xpos=0, ypos=0, zpos=0):
    ''' Simulate nIsochromats magnetization vectors per component with uniform distribution of Larmor frequencies.

    Args:
        config: configuration dictionary.
        component:  component specification from config.
        Meq:    equilibrium magnetization along z axis.
        M0:     initial state of magnetization vector, numpy array of size 3.
        xpos:   position of magnetization vector along x gradient.
        ypos:   position of magnetization vector along y gradient.
        zpos:   position of magnetization vector along z gradient.
        
    Returns:
        component magnetization vectors over time, numpy array of size [nIsochromats, 3, nFrames]

    '''
    if not M0:
        M0 = [0, 0, Meq] # Default initial state is equilibrium magnetization
    # Shifts in ppm for dephasing vectors:
    isochromats = [(2*i+1-config['nIsochromats'])/2*config['isochromatStep']+component['CS'] for i in range(0, config['nIsochromats'])]
    comp = np.empty((config['nIsochromats'],3,len(config['t'])))

    for m, isochromat in enumerate(isochromats):
        w = config['w0']*isochromat*1e-6  # Demodulated frequency [kRad / s]
        comp[m,:,:] = applyPulseSeq(config, Meq, M0, w, component['T1'], component['T2'], xpos, ypos, zpos)
    return comp


def getText(config):
    ''' Get opacity and display text pulseSeq event text flashes in 3D plot and store in config.
    
    Args:
        config: configuration dictionary.
        
    '''

    # Setup display text related to pulseSeq events:
    config['RFtext'] = np.full([len(config['t'])], '', dtype=object)
    config['Gtext'] = np.full([len(config['t'])], '', dtype=object)
    config['spoiltext'] = 'spoiler'
    config['RFalpha'] = np.zeros([len(config['t'])])   
    config['Galpha'] = np.zeros([len(config['t'])])
    config['spoilAlpha'] = np.zeros([len(config['t'])])   
        
    for rep in range(config['nTR']):
        TRstartFrame = rep * config['nFramesPerTR']
        for i, event in enumerate(config['events']):
            firstFrame, lastFrame = getEventFrames(config, i)        
            firstFrame += TRstartFrame
            lastFrame += TRstartFrame

            if 'RFtext' in event:
                config['RFtext'][firstFrame:] = event['RFtext']
                config['RFalpha'][firstFrame:lastFrame+1] = 1.0
            if any('{}text'.format(g) in event for g in ['Gx', 'Gy', 'Gz']): # gradient event
                Gtext = ''
                for g in ['Gx', 'Gy', 'Gz']:
                    if '{}text'.format(g) in event:
                        Gtext += '  ' + event['{}text'.format(g)]
                config['Gtext'][firstFrame:] = Gtext
                config['Galpha'][firstFrame:lastFrame+1] = 1.0
            if 'spoil' in event and event['spoil']:
                config['spoilAlpha'][firstFrame] = 1.0


def roundEventTime(time):
    return np.round(time, decimals=6) # nanosecond precision should be enough


def addEventsToTimeVector(t, pulseSeq):
    t = list(t)
    for event in pulseSeq:
        t.append(event['t'])
    return np.unique(roundEventTime(np.array(t)))


def calculateFA(B1vector, dwell):
    FA = 0
    for B1 in B1vector:
        FA += 360*(dwell * gyro * np.real(B1) * 1e-6)
    return FA


def checkPulseSeq(config):
    ''' Check/verify pulse sequence given by config.
    
    Args:
        config: configuration dictionary.
    '''

    instantDuration = 1e-3 # 1 us

    if 'pulseSeq' not in config:
        config['pulseSeq'] = []
    allowedKeys = ['t', 'spoil', 'dur', 'FA', 'B1', 'phase', 'Gx', 'Gy', 'Gz']
    for event in config['pulseSeq']:
        for item in event.keys(): # allowed keys
            if item not in allowedKeys:
                raise Exception('PulseSeq key "{}" not supported'.format(item))
        if not 't' in event:
            raise Exception('All pulseSeq events must have an event time "t"')
        else:
            event['t'] = roundEventTime(event['t'])
        if not any(key in event for key in ['FA', 'B1', 'Gx', 'Gy', 'Gz', 'spoil']):
            raise Exception('Empty events not allowed')
        if roundEventTime(event['t']) > config['TR']:
            raise Exception('pulseSeq event t exceeds TR')
        if 'dur' in event and roundEventTime(event['t'] + event['dur']) > config['TR']:
            raise Exception('pulseSeq event t+dur exceeds TR')
        if 'spoil' in event: # Spoiler event
            if not event['spoil']:
                raise Exception('Spoiler event must have spoil: true')
            if any([key not in ['t', 'spoil'] for key in event]):
                raise Exception('Spoiler event should only have event time t and spoil: true')
            event['spoiltext'] = 'spoiler'
        if 'FA' in event or 'B1' in event: # RF-pulse event (possibly with gradient)

            # combinations not allowed:
            if 'B1' in event and not any([key in event for key in ['dur', 'dwell']]):
                raise Exception('RF-pulse must provide "dur" or "dwell" along with "B1"')
            if 'dwell' in event:
                if 'B1' not in event or type(event['B1']) is not list:
                    raise Exception('"dwell" should be used for RF-pulses with "B1" as a list')
            if all([key in event for key in ['dur', 'dwell']]):
                raise Exception('RF-pulse cannot have both "dur" and "dwell"')
            
            # calculate duration:
            if ('B1' not in event and 'dur' not in event) or ('dur' in event and event['dur']==0):
                event['dur'] = instantDuration # handle "instant" RF pulse
            if 'dwell' in event:
                event['dur'] = len(event['B1']) * event['dwell']
            
            if 'B1' in event and type(event['B1']) is not list:
                event['B1'] = [event['B1']] # make it a list
            
            # calculate dwell time:
            if 'dwell' not in event:
                if 'B1' in event:
                    event['dwell'] = event['dur'] / len(event['B1'])
                else:
                    event['dwell'] = event['dur']
            
            # calculate FA prescribed by B1
            if 'B1' in event:
                calcFA = calculateFA(event['B1'], event['dwell'])

            # calculate B1 or scale it to get prescribed FA
            if 'FA' in event:
                if 'B1' not in event:
                    event['B1'] = [event['FA']/(event['dur'] * 360 * gyro * 1e-6)]
                else:
                    event['B1'] = [B1 * event['FA'] / calcFA for B1 in event['B1']] # scale B1 to get prescribed FA
            else:
                event['FA'] = calcFA

            # make B1 an array
            event['B1'] = np.array(event['B1'])

            if 'phase' in event: # Add phase to B1 if provided
                event['B1'] = event['B1'] * np.exp(1j * np.radians(event['phase']))
            event['w1'] = [2 * np.pi * gyro * B1 * 1e-6 for B1 in event['B1']] # kRad / s
            event['RFtext'] = str(int(abs(event['FA'])))+u'\N{DEGREE SIGN}'+'-pulse'
        if any(key in event for key in ['Gx', 'Gy', 'Gz']): # Gradient (no RF)
            if not ('dur' in event and event['dur']>0):
                raise Exception('Gradient must have a specified duration>0 (dur [ms])')
            if 'FA' not in event and 'phase' in event:
                raise Exception('Gradient event should have no phase')
            for g in ['Gx', 'Gy', 'Gz']:
                if g in event:
                    try:
                        event[g] = float(event[g])
                    except ValueError:
                        print('{} must be a number'.format(g))
                        raise ValueError
                    
                    event['{}text'.format(g)] = '{}: {} mT/m'.format(g, event[g])
    
    # Sort pulseSeq according to event time
    config['pulseSeq'] = sorted(config['pulseSeq'], key=lambda event: event['t'])

    # split any pulseSeq events with array values into separate events
    config['separatedPulseSeq'] = []
    for event in config['pulseSeq']:
        if 'dwell' in event:
            for i, t in enumerate(np.arange(event['t'], event['t'] + event['dur'], event['dwell'])):
                subEvent = {'t': roundEventTime(t), 'dur': event['dwell']}
                if i==0 and spoil in event:
                    subEvent['spoil'] = event['spoil']
                if 'RFtext' in event:
                    subEvent['RFtext'] = event['RFtext']
                for key in ['w1', 'Gx', 'Gy', 'Gz']:
                    if key in event:
                        if type(event[key]) is list:
                            subEvent[key] = event[key][i]
                        else:
                            subEvent[key] = event[key]
                config['separatedPulseSeq'].append(subEvent)
        else:
            config['separatedPulseSeq'].append(event)

    # Sort separatedPulseSeq according to event time
    config['separatedPulseSeq'] = sorted(config['separatedPulseSeq'], key=lambda event: event['t'])


def emptyEvent():
    return {'w1': 0, 'Gx': 0, 'Gy': 0, 'Gz': 0, 'spoil': False}


def mergeEvent(event, event2merge, t):
    for channel in ['w1', 'Gx', 'Gy', 'Gz']:
        if channel in event2merge:
            event[channel] += event2merge[channel]
    for text in ['RFtext', 'Gxtext', 'Gytext', 'Gztext', 'spoilText']:
        if text in event2merge:
            event[text] = event2merge[text]
    if 'spoil' in event2merge:
        event['spoil'] = True
    else:
        event['spoil'] = False
    event['t'] = t
    return event


def detachEvent(event, event2detach, t):
    for channel in ['w1', 'Gx', 'Gy', 'Gz']:
        if channel in event2detach:
            event[channel] -= event2detach[channel]
    for text in ['RFtext', 'Gxtext', 'Gytext', 'Gztext', 'spoilText']:
        if text in event and text in event2detach and event[text]==event2detach[text]:
            del event[text]
    event['t'] = t
    return event


def getPrescribedTimeVector(config, nTR):
    dt = 1e3/config['fps']*config['speed'] # Animation time resolution [msec]
    kernelTime = np.arange(0, config['TR'], dt) # kernel time vector [msec]

    slowRFpulses = True
    if slowRFpulses:
        for event in config['pulseSeq']:
            if 'FA' in event:
                dt = event['dur'] / config['fps'] * 90 / abs(event['FA']) # dt [msec] that gives one sec per 90 flip
                kernelTime = np.concatenate((kernelTime, np.arange(event['t'], event['t'] + event['dur'], dt)), axis=None)
    
    t = np.array([])
    for rep in range(nTR): # Repeat time vector for each TR
        t = np.concatenate((t, kernelTime + rep * config['TR']), axis=None)
    return np.unique(roundEventTime(t))


def setupPulseSeq(config):
    ''' Check and setup pulse sequence given by config. Set clock and store in config.
    
    Args:
        config: configuration dictionary.
        
    '''

    checkPulseSeq(config)

    # Create non-overlapping events, each with constant w1, Gx, Gy, Gz, including empty "relaxation" events
    config['events'] = []
    ongoingEvents = []
    newEvent = emptyEvent() # Start with empty "relaxation event"
    newEvent['t'] = 0
    for i, event in enumerate(config['separatedPulseSeq']):
        # Merge any events starting simultaneously:
        if event['t']==newEvent['t']:
            newEvent = mergeEvent(newEvent, event, event['t'])
        else:
            config['events'].append(dict(newEvent))
            newEvent = mergeEvent(newEvent, event, event['t'])
        if 'dur' in event: # event is ongoing unless no 'dur', i.e. spoiler event
                ongoingEvents.append(event)
                # sort ongoing events according to event end time:
                sorted(ongoingEvents, key=lambda event: event['t'] + event['dur'], reverse=False)
        if event is config['separatedPulseSeq'][-1]:
            nextEventTime = config['TR']
        else:
            nextEventTime = config['separatedPulseSeq'][i+1]['t']
        for stoppingEvent in [event for event in ongoingEvents[::-1] if roundEventTime(event['t'] + event['dur']) <= nextEventTime]:
            config['events'].append(dict(newEvent))
            newEvent = detachEvent(newEvent, stoppingEvent, roundEventTime(stoppingEvent['t'] + stoppingEvent['dur']))
            ongoingEvents.pop()
    config['events'].append(dict(newEvent))

    # Set clock vector
    config['kernelClock'] = getPrescribedTimeVector(config, 1)
    config['kernelClock'] = addEventsToTimeVector(config['kernelClock'], config['events'])
    if config['kernelClock'][-1] == config['TR']:
        config['kernelClock'] = config['kernelClock'][:-1]
    config['nFramesPerTR'] = len(config['kernelClock'])
    config['t'] = np.array([])
    for rep in range(config['nTR']): # Repeat time vector for each TR
        config['t'] = np.concatenate((config['t'], roundEventTime(config['kernelClock'] + rep * config['TR'])), axis=None)
    config['t'] = np.concatenate((config['t'], roundEventTime(config['nTR'] * config['TR'])), axis=None) # Add end time to time vector
    config['kernelClock'] = np.concatenate((config['kernelClock'], config['TR']), axis=None) # Add end time to kernel clock
    

def arrangeLocations(slices, config, key='locations'):
    ''' Check and setup locations or M0. Set nx, ny, and nz and store in config.
    
    Args:
        slices: (nested) list of M0 or locations (spatial distribution of Meq).
        config: configuration dictionary.
        key:    pass 'locations' for Meq distribution, and 'M0' for M0 distribution.
        
    '''
    if key not in ['M0', 'locations']:
        raise Exception('Argument "key" must be "locations" or "M0", not {}'.format(key))
    if not isinstance(slices, list):
        raise Exception('Expected list in config "{}", not {}'.format(key, type(slices)))
    if not isinstance(slices[0], list):
        slices = [slices]
    if not isinstance(slices[0][0], list):
        slices = [slices]
    if key=='M0' and not isinstance(slices[0][0][0], list):
        slices = [slices]
    if 'nz' not in config:
        config['nz'] = len(slices)
    elif len(slices)!=config['nz']:
        raise Exception('Config "{}": number of slices do not match'.format(key))
    if 'ny' not in config:
        config['ny'] = len(slices[0])
    elif len(slices[0])!=config['ny']:
        raise Exception('Config "{}": number of rows do not match'.format(key))
    if 'nx' not in config:
        config['nx'] = len(slices[0][0])
    elif len(slices[0][0])!=config['nx']:
        raise Exception('Config "{}": number of elements do not match'.format(key))
    if key=='M0' and len(slices[0][0][0])!=3:
        raise Exception('Config "{}": inner dimension must be of length 3'.format(key))
    return slices


def checkConfig(config):
    ''' Check and setup config.
    
    Args:
        config: configuration dictionary.
        
    '''
    if any([key not in config for key in ['TR', 'B0', 'speed', 'output']]):
        raise Exception('Config must contain "TR", "B0", "speed", and "output"')
    if 'title' not in config:
        config['title'] = ''
    config['TR'] = roundEventTime(config['TR'])
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
    
    config['w0'] = 2*np.pi*gyro*config['B0'] # Larmor frequency [kRad/s]

    setupPulseSeq(config)

    ### Arrange locations ###
    if not 'collapseLocations' in config:
        config['collapseLocations'] = False
    if not 'locations' in config:
        config['locations'] = arrangeLocations([[[1]]], config)
    else:
        if isinstance(config['locations'], dict):
            for comp in iter(config['locations']):
                config['locations'][comp] = arrangeLocations(config['locations'][comp], config)
        elif isinstance(config['locations'], list):
            locs = config['locations']
            config['locations'] = {}
            for comp in [n['name'] for n in config['components']]:
                config['locations'][comp] = arrangeLocations(locs, config)
        else:
            raise Exception('Config "locations" should be list or components dict')
    for (FOV, n) in [('FOVx', 'nx'), ('FOVy', 'ny'), ('FOVz', 'nz')]:
        if FOV not in config:
            config[FOV] = config[n]*config['locSpacing'] #FOV in m
    if 'M0' in config:
        if isinstance(config['M0'], dict):
            for comp in iter(config['M0']):
                config['M0'][comp] = arrangeLocations(config['M0'][comp], config, 'M0')
        elif isinstance(config['M0'], list):
            M0 = config['M0']
            config['M0'] = {}
            for comp in [n['name'] for n in config['components']]:
                config['M0'][comp] = arrangeLocations(M0, config, 'M0')
        else:
            raise Exception('Config "M0" should be list or components dict')
    
    # check output
    for output in config['output']:
        if 'dpi' not in output:
            output['dpi'] = 100
        if output['type']=='3D':
            if 'drawAxes' not in output:
                output['drawAxes'] = config['nx']*config['ny']*config['nz'] == 1
    if 'background' not in config:
        config['background'] = {}
    if 'color' not in config['background']:
        config['background']['color'] = 'black'
    if config['background']['color'] not in ['black', 'white']:
        raise Exception('Only "black" and "white" supported as background colors')


def rotMatrix(angle, axis):
    '''Get 3D rotation matrix.
    
    Args:
        angle:  rotation angle in radians.
        axis:   axis of rotation (0, 1, or 2).
        
    Returns:
        rotation matrix, numpy array of size [3, 3]

    '''
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[1,0,0], [0,c,-s], [0,s,c]])
    return np.roll(np.roll(R, axis, axis=0), axis, axis=1)
    

def spherical2cartesian(spherical):
    '''Convert 3D vector from spherical to Cartesian coordinates.
    
    Args:
        spherical: 3-tuple holding vector length, polar, and azimuthal angle
        
    Returns:
        Cartesian vector, list of size 3

    '''
    r, polar, azim = spherical
    M = np.dot(np.dot(np.array([0, 0, r]), rotMatrix(np.radians(azim), 1)), rotMatrix(np.radians(polar), 2))
    return list(M)


def resampleOnPrescribedTimeFrames(vectors, config):
    config['tFrames'] = getPrescribedTimeVector(config, config['nTR'])
    newShape = list(vectors.shape)
    newShape[6] = len(config['tFrames'])
    resampledVectors = np.zeros(newShape)
    for x in range(newShape[0]):
        for y in range(newShape[1]):
            for z in range(newShape[2]):
                for c in range(newShape[3]):
                    for i in range(newShape[4]):
                        for dim in range(newShape[5]):
                            resampledVectors[x,y,z,c,i,dim,:] = np.interp(config['tFrames'], config['t'], vectors[x,y,z,c,i,dim,:])
    # resample text alpha channels:
    for channel in ['RFalpha', 'Galpha', 'spoilAlpha']:
        alphaVector = np.zeros([len(config['tFrames'])])
        #config[channel] = np.interp(config['tFrames'], config['t'], config[channel])
        for i in range(len(alphaVector)):
            if i == len(alphaVector)-1:
                ks = np.where(config['t']>=config['tFrames'][i])[0]
            else:
                ks = np.where(np.logical_and(config['t']>=config['tFrames'][i], config['t']<config['tFrames'][i+1]))[0]
            alphaVector[i] = np.max(config[channel][ks])
        config[channel] = alphaVector
    # resample text:
    for text in ['RFtext', 'Gtext']:
        textVector = np.full([len(config['tFrames'])], '', dtype=object)
        for i in range(len(textVector)):
            k = np.where(config['t']>=config['tFrames'][i])[0][0]
            textVector[i] = config[text][k]
        config[text] = textVector
    return resampledVectors


def fadeTextFlashes(config, fadeTime=1.0):
    ''' Modify text alpha channels such that the text flashes fade
    
    Args:
        config:     configuration dictionary.
        fadeTime:   time of fade in seconds   

    '''
    decay = 1.0/(config['fps'] * fadeTime) # alpha decrease per frame
    for channel in ['RFalpha', 'Galpha', 'spoilAlpha']:
        for i in range(1, len(config[channel])):
            if config[channel][i]==0:
                config[channel][i] = max(0, config[channel][i-1]-decay)


def run(configFile, leapFactor=1, gifWriter='ffmpeg'):
    ''' Main program. Read and setup config, simulate magnetization vectors and write animated gif.
        
    Args:
        configFile: YAML file specifying configuration.
        leapFactor: Skip frame factor for faster processing and smaller filesize.
        gifWriter:  external program to write gif. Must be "ffmpeg" or "imagemagick"/"convert".
        
    '''
    # Check if gifWriter exists
    gifWriter = gifWriter.lower()
    if gifWriter == 'ffmpeg':
        if not shutil.which('ffmpeg'):
            raise Exception('FFMPEG not found')
    elif gifWriter in ['imagemagick', 'convert']:
        if not shutil.which('convert'):
            raise Exception('ImageMagick (convert) not found')
    else:
        raise Exception('Argument gifWriter must be "ffmpeg" or "imagemagick"/"convert"')

    # Set global constants
    global gyro
    gyro = 42577.			# Gyromagnetic ratio [kHz/T]

    # Read configuration file
    with open(configFile, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise Exception('Error reading config file') from exc
    
    ### Setup config correctly ###
    checkConfig(config)

    if config['background']['color'] == 'black':
        for i in ['bg', 'axis', 'text', 'circle']:
            colors[i][:3] = list(map(lambda x: 1-x, colors[i][:3]))

    ### Simulate ###
    vectors = np.empty((config['nx'],config['ny'],config['nz'],config['nComps'],config['nIsochromats'],3,len(config['t'])))
    for z in range(config['nz']):
        for y in range(config['ny']):
            for x in range(config['nx']):
                for c, component in enumerate(config['components']):
                    if component['name'] in config['locations']:
                        try:
                            Meq = config['locations'][component['name']][z][y][x]
                        except:
                            raise Exception('Is the "location" matrix shape equal for all components?')
                    elif isinstance(config['locations'], list):
                        Meq = config['locations'][z][y][x]
                    else:
                        Meq = 0.0
                    if 'M0' in config and component['name'] in config['M0']:
                        try:
                            M0 = spherical2cartesian(config['M0'][component['name']][z][y][x])
                        except:
                            raise Exception('Is the "M0" matrix shape equal for all components?')
                    elif 'M0' in config and isinstance(config['M0'], list):
                        M0 = spherical2cartesian(config['M0'][z][y][x])
                    else:
                        M0 = None
                    xpos = (x+.5-config['nx']/2)*config['locSpacing']
                    ypos = (y+.5-config['ny']/2)*config['locSpacing']
                    zpos = (z+.5-config['nz']/2)*config['locSpacing']
                    vectors[x,y,z,c,:,:,:] = simulateComponent(config, component, Meq, M0, xpos, ypos, zpos)

    ### Animate ###
    getText(config) # prepare text flashes for 3D plot
    vectors = resampleOnPrescribedTimeFrames(vectors, config)
    fadeTextFlashes(config)
    delay = int(100/config['fps']*leapFactor)  # Delay between frames in ticks of 1/100 sec

    outdir = './out'
    for output in config['output']:
        if output['file']:
            if output['type'] in ['xy', 'z']:
                signal = np.sum(vectors, (0,1,2,4)) # sum over space and isochromats
                if 'normalize' in output and output['normalize']:
                    for c, comp in enumerate([n['name'] for n in config['components']]):
                        signal[c,:] /= np.sum(config['locations'][comp])
                signal /= np.max(np.abs(signal)) # scale signal relative to maximum
                if 'scale' in output:
                    signal *= output['scale']
            if gifWriter == 'ffmpeg':
                ffmpegWriter = FFMPEGwriter.FFMPEGwriter(config['fps'])
            else:
                tmpdir = './tmp'
                if os.path.isdir(tmpdir):
                    rmTmpDir = input('Temporary folder "{}" already exists. Delete(Y/N)?'.format(tmpdir))
                    if rmTmpDir.upper() == 'Y':
                        shutil.rmtree(tmpdir)
                    else:
                        raise Exception('No files written.')
                os.makedirs(tmpdir, exist_ok=True)
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, output['file'])
            for frame in range(0, len(config['tFrames']), leapFactor):
                # Use only every leapFactor frame in animation
                if output['type'] == '3D':
                    fig = plotFrame3D(config, vectors, frame, output)
                elif output['type'] == 'kspace':
                    fig = plotFrameKspace(config, frame, output)
                elif output['type'] == 'psd':
                    fig = plotFramePSD(config, frame, output)
                elif output['type'] in ['xy', 'z']:
                    fig = plotFrameMT(config, signal, frame, output)
                plt.draw()
                if gifWriter == 'ffmpeg':
                    ffmpegWriter.addFrame(fig)
                else: # use imagemagick: save frames temporarily 
                    file = os.path.join(tmpdir, '{}.png'.format(str(frame).zfill(4)))
                    print('Saving frame {}/{} as "{}"'.format(frame+1, config['nFrames'], file))
                    plt.savefig(file, facecolor=plt.gcf().get_facecolor())
                plt.close()
            if gifWriter == 'ffmpeg':
                ffmpegWriter.write(outfile)
            else: # use imagemagick
                print('Creating animated gif "{}"'.format(outfile))
                compress = '-layers Optimize'
                os.system(('convert {} -delay {} {}/*png {}'.format(compress, delay, tmpdir, outfile)))
                shutil.rmtree(tmpdir)


def parseAndRun():
    ''' Command line parser. Parse command line and run main program. '''

    # Initiate command line parser
    parser = argparse.ArgumentParser(description='Simulate magnetization vectors using Bloch equations and create animated gif')
    parser.add_argument('--configFile', '-c',
                        help="Name of configuration text file",
                        type=str,
                        default='')
    parser.add_argument('--leapFactor', '-l',
                        help="Leap factor for smaller filesize and fewer frames per second",
                        type=int,
                        default=1)

    # Parse command line
    args = parser.parse_args()

    # Run main program
    run(args.configFile, args.leapFactor)

if __name__ == '__main__':
    parseAndRun()
