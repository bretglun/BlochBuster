#!/usr/bin/env python3

# -*- coding: utf-8 -*-
# Copyright (c) 2017-2023 Johan Berglund
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
from numbers import Number
import scipy.integrate as integrate
from scipy.stats import norm
from pathlib import Path
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
            'kSpacePos': [1, .5, 0],
            'B1vector':   [1, 1, 0]
            }

class Arrow3D(FancyArrowPatch):
    '''Matplotlib FancyArrowPatch for 3D rendering.'''
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        self.do_3d_projection()
        FancyArrowPatch.draw(self, renderer)
    
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def figureSize(dpi, width, height):
    '''Calculates a frame size with even number of pixels in height and widh
    Args:
        dpi: pixels per inch
        width: width in inches
        height: height in inches

    Returns:
        pixelsWidth, pixelsHeight
    '''
    pxw = width * dpi
    pxh = height * dpi
    pxh += pxh % 2
    pxw += pxh % 2
    return (pxw, pxh)


def plotBlackWhiteSphere(ax, M, pos, R=.4, rotationAxis='y', nPoints=36):
    '''Plot phase of transverse magnetization as a sphere with a black and a white half
       Following Plewes and Kucharczyk, JMRI 35(5): 1038-54, 2012.
    Args:
        ax: matplotlib axes for plotting
        M: magnetization vector [Mx, My, Mz]
        pos: position of magnetization vector [x, y, z]
        R: sphere radius in units of locSpacing per unit magnitude
        rotationAxis: axis around which the sphere rotates ('x', 'y', or 'z')
        nPoints: number of points along sphere circumference
    '''
    phase = np.arctan2(M[1], M[0])
    m = np.linalg.norm(M)
    if m<.05: return
    phi, theta = np.mgrid[0.0:2.0*np.pi:nPoints*1j, 0.0:np.pi:nPoints/2*1j]
    dx = m * R * np.sin(theta) * np.cos(phi)
    dy = m * R * np.sin(theta) * np.sin(phi)
    dz = m * R * np.cos(theta)

    colors = np.full_like(dx, 'white', dtype=object) # define colors for the two sides
    colors[dy < 0] = 'dimgrey' # divide across y-axis

    dx, dy = dx * np.cos(phase) - dy * np.sin(phase), dx * np.sin(phase) + dy * np.cos(phase) # rotate
    if rotationAxis=='x':
        dx, dz = dz, dx
    elif rotationAxis=='y':
        dy, dz = dz, dy
    ax.plot_surface(dx+pos[0], dy-pos[1], dz-pos[2], facecolors=colors, shade=True)


def plotFrame3D(config, vectors, B1vector, frame, output):
    '''Creates a plot of magnetization vectors in a 3D view.
    
    Args:
        config: configuration dictionary.
        vectors:    numpy array of size [nx, ny, nz, nComps, nIsochromats, 6, nFrames].
        B1vector:   complex numpy array of size [nFrames].
        frame:      which frame to plot.
        output:     specification of desired output (dictionary from config).

    Returns:
        plot figure.

    '''
    nx, ny, nz, nComps, nIsoc = vectors.shape[:5]

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
    pxw, pxh = figureSize(dpi=output['dpi'], width=figSize, height=figSize * aspect)
    fig = plt.figure(figsize=(pxw/output['dpi'], pxh/output['dpi']), dpi=output['dpi'])
    axLimit = max(nx,ny,nz)/2+.5
    if config['collapseLocations']:
        axLimit = 1.0
    ax = plt.axes(projection='3d', xlim=(-axLimit,axLimit), ylim=(-axLimit,axLimit), zlim=(-axLimit,axLimit), fc=colors['bg'])
    
    if nx*ny*nz>1 and not config['collapseLocations']:
        ax.view_init(azim=output['azimuth'], elev=output['elevation'])
    ax.set_axis_off()
    width = 1.6 # to get tight cropping
    height = width/aspect
    left = (1-width)/2
    bottom = (1-height)/2
    if nx*ny*nz==1 or config['collapseLocations']: # shift to fit legend
        left -= .02
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
        ax.text(1.08, 0, 0, r'$x{}$'.format('' if 'rotate' in output else '^\prime'), horizontalalignment='center', color=colors['text'])
        ax.plot([0, 0], [-1, 1], [0, 0], c=colors['axis'], zorder=-1)  # y-axis
        ax.text(0, 1.12, 0, r'$y{}$'.format('' if 'rotate' in output else '^\prime'), horizontalalignment='center', color=colors['text'])
        ax.plot([0, 0], [0, 0], [-1, 1], c=colors['axis'], zorder=-1)  # z-axis
        ax.text(0, 0, 1.05, r'$z$', horizontalalignment='center', color=colors['text'])

    # Draw title:
    fig.text(.5, 1, config['title'], fontsize=14, horizontalalignment='center', verticalalignment='top', color=colors['text'])

    # Draw time
    time = config['tFrames'][frame%(len(config['t'])-1)] # frame time [msec]
    time_text = fig.text(0, 0, 'time = %.1f msec' % (time), color=colors['text'], verticalalignment='bottom')

    # TODO: put isochromats in this order from start
    order = [int((nIsoc-1)/2-abs(m-(nIsoc-1)/2)) for m in range(nIsoc)]
    arrowheadThres = 0.075 * axLimit # threshold on vector magnitude for arrowhead shrinking
    projection = np.array([np.cos(np.deg2rad(ax.azim)) * np.cos(np.deg2rad(ax.elev)),
                           np.sin(np.deg2rad(ax.azim)) * np.cos(np.deg2rad(ax.elev)),
                           np.sin(np.deg2rad(ax.elev))])
    if 'rotate' in output:
        rotFreq = output['rotate'] * 1e-3 # coordinate system rotation relative resonance frequency [kHz]
        rotMat = rotMatrix(2 * np.pi * rotFreq * time, 2) # rotation matrix for rotating coordinate system

    # Draw magnetization vectors
    pos = [0,0,0]
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                for c in range(nComps):
                    for m in range(nIsoc):
                        col = colors['comps'][(c) % len(colors['comps'])]
                        M = vectors[x,y,z,c,m,:3,frame]
                        if not config['collapseLocations']:
                            pos = vectors[x,y,z,c,m,3:,frame]/config['locSpacing']
                        if 'rotate' in output:
                            M = np.dot(M, rotMat) # rotate vector relative to coordinate system                        
                        if 'spheres' in output and output['spheres']:
                            
                            plotBlackWhiteSphere(ax, M[:3], pos[:3])
                        else:
                            Mnorm = np.sqrt((np.linalg.norm(M)**2 - np.dot(M, projection)**2)) # vector norm in camera projection
                            arrowScale = 20 if Mnorm > arrowheadThres else 20 * Mnorm/arrowheadThres # Shrink arrowhead for short arrows
                            alpha = 1.-2*np.abs((m+.5)/nIsoc-.5)
                            ax.add_artist(Arrow3D(  [pos[0], pos[0]+M[0]], 
                                                    [-pos[1], -pos[1]+M[1]],
                                                    [-pos[2], -pos[2]+M[2]], 
                                                    mutation_scale=arrowScale,
                                                    arrowstyle='-|>', shrinkA=0, shrinkB=0, lw=2,
                                                    color=col, alpha=alpha, 
                                                    zorder=order[m]+nIsoc*int(100*(1-Mnorm))))
    
    # Draw B1 vector
    if config['plotB1']:
        M = np.array([np.imag(B1vector[frame]), -np.real(B1vector[frame]), 0])
        if 'rotate' in output: 
            M = np.dot(M, rotMat) # rotate vector relative to coordinate system                        
        Mnorm = np.sqrt((np.linalg.norm(M)**2 - np.dot(M, projection)**2)) # vector norm in camera projection
        arrowScale = 40 if Mnorm > (arrowheadThres*4) else 40 * Mnorm/(arrowheadThres*4) # Shrink arrowhead for short arrows
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    pos = [ (x+.5-config['nx']/2),
                            (y+.5-config['ny']/2),
                            (z+.5-config['nz']/2) ]
                    ax.add_artist(Arrow3D(  [pos[0], pos[0] + M[0]], 
                                            [-pos[1], -pos[1] + M[1]],
                                            [-pos[2], -pos[2]] + M[2], 
                                            mutation_scale = arrowScale,
                                            arrowstyle='->', shrinkA=0, shrinkB=0, lw=4,
                                            color=colors['B1vector']))

    # Draw "spoiler" and "FA-pulse" text
    fig.text(1, .94, config['RFtext'][frame], fontsize=14, alpha=config['RFalpha'][frame],
            color=colors['RFtext'], horizontalalignment='right', verticalalignment='top')
    fig.text(1, .88, config['Gtext'][frame], fontsize=14, alpha=config['Galpha'][frame],
            color=colors['Gtext'], horizontalalignment='right', verticalalignment='top')
    fig.text(1, .82, config['spoiltext'], fontsize=14, alpha=config['spoilAlpha'][frame],
            color=colors['spoilText'], horizontalalignment='right', verticalalignment='top')

    # Draw legend:
    if 'legend' in output and output['legend']:
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
        signal: numpy array of size [nComps, 6, nFrames].
        frame:  which frame to plot up to.
        output: specification of desired output (dictionary from config).

    Returns:
        plot figure.

    '''
    if output['type'] not in ['xy', 'z']:
        raise Exception('output "type" must be 3D, kspace, psd, xy (transversal) or z (longitudinal)')

    # create diagram
    xmin, xmax = output['tRange']
    
    if output['type'] == 'xy':
        if 'abs' in output and not output['abs']:
            ymin, ymax = -1, 1
        else:
            ymin, ymax = 0, 1
    elif output['type'] == 'z':
        ymin, ymax = -1, 1

    pxw, pxh = figureSize(dpi=output['dpi'], width=5, height=2.7)
    fig = plt.figure(figsize=(pxw/output['dpi'], pxh/output['dpi']), facecolor=colors['bg'], dpi=output['dpi'])
    ax = fig.add_subplot(xlim=(xmin, xmax), ylim=(ymin, ymax), fc=colors['bg'])
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
    ax.arrow(xmin, ymin, 0, (ymax-ymin)*1.05, fc=colors['text'], ec=colors['text'], lw=1, head_width=yhw, head_length=yhl, clip_on=False, zorder=100)
    
    # Draw magnetization vectors
    nComps = signal.shape[0]
    if output['type'] == 'xy':
        if not output['sum']:
            for c in range(nComps):
                col = colors['comps'][c % len(colors['comps'])]
                if 'abs' in output and not output['abs']: # real and imag part of transversal magnetization
                    ax.plot(config['tFrames'][:frame+1], signal[c,0,:frame+1], '-', lw=2, color=col)
                    col = colors['comps'][c+nComps+1 % len(colors['comps'])]
                    ax.plot(config['tFrames'][:frame+1], signal[c,1,:frame+1], '-', lw=2, color=col)
                else: # absolute value of transversal magnetization
                    ax.plot(config['tFrames'][:frame+1], np.linalg.norm(signal[c,:2,:frame+1], axis=0), '-', lw=2, color=col)
        if output['sum'] or all(key in [comp['name'] for comp in config['components']] for key in ['water', 'fat']):
            # plot sum component also if both water and fat (special case)
            col = colors['comps'][nComps % len(colors['comps'])]
            if 'abs' in output and not output['abs']: # real and imag part of transversal magnetization
                ax.plot(config['tFrames'][:frame+1], np.mean(signal[:,0,:frame+1],0), '-', lw=2, color=col)
                col = colors['comps'][(2*nComps+1) % len(colors['comps'])]
                ax.plot(config['tFrames'][:frame+1], np.mean(signal[:,1,:frame+1],0), '-', lw=2, color=col)
            else: # absolute value of transversal magnetization
                ax.plot(config['tFrames'][:frame+1], np.linalg.norm(np.mean(signal[:,:2,:frame+1],0), axis=0), '-', lw=2, color=col)

    elif output['type'] == 'z':
        for c in range(nComps):
            col = colors['comps'][(c) % len(colors['comps'])]
            ax.plot(config['tFrames'][:frame+1], signal[c,2,:frame+1], '-', lw=2, color=col)

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
    pxw, pxh = figureSize(dpi=output['dpi'], width=5, height=5)
    fig = plt.figure(figsize=(pxw/output['dpi'], pxh/output['dpi']), facecolor=colors['bg'], dpi=output['dpi'])
    ax = fig.add_subplot(xlim=(xmin, xmax), ylim=(ymin, ymax), fc=colors['bg'])
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
        xmin, xmax = output['tRange']
        ymin, ymax = 0, 5
        pxw, pxh = figureSize(dpi=output['dpi'], width=5, height=5)
        fig = plt.figure(figsize=(pxw/output['dpi'], pxh/output['dpi']), facecolor=colors['bg'], dpi=output['dpi'])
        ax = fig.add_subplot(xlim=(xmin, xmax), ylim=(ymin, ymax), fc=colors['bg'])
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
        
        boards = {'w1': {'ypos': 4}, 'Gx': {'ypos': 3}, 'Gy': {'ypos': 2}, 'Gz': {'ypos': 1}}
        for board in boards:
            boards[board]['signal'] = [0]
        t = [0]
        for event in config['events']:
            for board in boards:
                boards[board]['signal'].append(boards[board]['signal'][-1]) # end of previous event:
                boards[board]['signal'].append(event[board]) # start of this event:
            t.append(event['t']) # end of previous event:
            t.append(event['t']) # start of this event:

        boards['w1']['scale'] = 0.48 / np.max([np.abs(w) for w in boards['w1']['signal'] if np.abs(w) < 50])
        if 'gmax' not in output:
            output['gmax'] = np.max(np.abs(np.concatenate((boards['Gx']['signal'], boards['Gy']['signal'], boards['Gz']['signal']))))
        boards['Gx']['scale'] = boards['Gy']['scale'] = boards['Gz']['scale'] = 0.48 / output['gmax']
                
        for board in ['w1', 'Gx', 'Gy', 'Gz']:
            ax.plot(t, boards[board]['ypos'] + np.array(boards[board]['signal']) * boards[board]['scale'], lw=1, color=colors['boards'][board])
            ax.plot([xmin, xmax], [boards[board]['ypos'], boards[board]['ypos']], color=colors['text'], lw=1, clip_on=False, zorder=100)
            ax.text(0, boards[board]['ypos'], board, fontsize=14,
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
        w:              Larmor frequency :math:`2\\pi\\gamma B_0` [kRad / s].
	    w1 (complex):   B1 rotation frequency :math:`2\\pi\\gamma B_1`  [kRad / s].
        T1:             longitudinal relaxation time.
        T2:             transverse relaxation time.
        M:              magnetization vector.
        Meq:            equilibrium magnetization.
        t:              time vector (needed for scipy.integrate.odeint).

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


def applyPulseSeq(config, Meq, M0, w, T1, T2, pos0, v, D):
    '''Simulate magnetization vector during nTR (+nDummies) applications of pulse sequence.
    
    Args:
        config: configuration dictionary.
        Meq:    equilibrium magnetization along z axis.
        M0:     initial state of magnetization vector, numpy array of size 3.
        w:      Larmor frequency :math:`2\\pi\\gamma B_0` [kRad/s].
        T1:     longitudinal relaxation time.
        T2:     transverse relaxation time.
        pos0:   position (x,y,z) of magnetization vector at t=0 [m].
        v:      velocity (x,y,z) of spins [mm/s]
        D:      diffusivity (x,y,z) of spins [:math:`mm^2/s`]
        
    Returns:
        magnetization vector over time, numpy array of size [6, nFrames]. 1:3 are magnetization, 4:6 are position

    '''
    M = np.zeros([len(config['t']), 3])
    M[0] = M0 # Initial state

    pos = np.tile(pos0, [len(config['t']), 1]) # initial position
    if np.linalg.norm(D) > 0: # diffusion contribution
        for frame in range(1,len(config['t'])):
            dt = config['t'][frame] - config['t'][frame-1]
            for dim in range(3):
                pos[frame][dim] = pos[frame-1][dim] + norm.rvs(scale=np.sqrt(D[dim]*dt*1e-9)) # standard deviation in meters
            if config['t'][frame]==0: # reset position for t=0
                pos[:frame+1] += np.tile(pos0 - pos[frame], [frame+1, 1])
    if np.linalg.norm(v) > 0: # velocity contribution
        pos += np.outer(config['t'], v) * 1e-6

    for rep in range(-config['nDummies'], config['nTR']): # dummy TRs get negative frame numbers
        TRstartFrame = rep * config['nFramesPerTR']

        for i, event in enumerate(config['events']):
            firstFrame, lastFrame = getEventFrames(config, i)
            firstFrame += TRstartFrame
            lastFrame += TRstartFrame

            M0 = M[firstFrame]

            if 'spoil' in event and event['spoil']: # Spoiler event
                M0 = spoil(M0)

            # frequency due to w plus any gradients 
            # (use position at firstFrame, i.e. approximate no motion during frame)
            wg = w  
            wg += 2*np.pi*gyro*event['Gx']*pos[firstFrame, 0]/1000 # [kRad/s]
            wg += 2*np.pi*gyro*event['Gy']*pos[firstFrame, 1]/1000 # [kRad/s]
            wg += 2*np.pi*gyro*event['Gz']*pos[firstFrame, 2]/1000 # [kRad/s]

            w1 = event['w1'] * np.exp(1j * np.radians(event['phase']))

            t = config['t'][firstFrame:lastFrame+1]
            if len(t)==0:
                raise Exception("Corrupt config['events']")
            M[firstFrame:lastFrame+1] = integrate.odeint(derivs, M0, t, args=(Meq, wg, w1, T1, T2)) # Solve Bloch equation

    return np.concatenate((M, pos),1).transpose()


def getB1vector(config):
    ''' Get transverse B1 vector over time, based on pulse sequence in config.

    Args:
        config: configuration dictionary.
        
    Returns:
        B1vector over time, complex numpy array of size [len(config['t'])].

    '''
    B1vector = np.empty((len(config['t'])), dtype=complex)
    
    for rep in range(-config['nDummies'], config['nTR']): # dummy TRs get negative frame numbers
        TRstartFrame = rep * config['nFramesPerTR']

        for i, event in enumerate(config['events']):
            firstFrame, lastFrame = getEventFrames(config, i)
            firstFrame += TRstartFrame
            lastFrame += TRstartFrame

            w1 = event['w1'] * np.exp(1j * np.radians(event['phase']))

            t = config['t'][firstFrame:lastFrame+1]
            if len(t)==0:
                raise Exception("Corrupt config['events']")
            B1vector[firstFrame:lastFrame+1] = w1
    B1vector /= max(abs(B1vector)) # normalize
    return B1vector


def simulateComponent(config, component, Meq, M0=None, pos=None):
    ''' Simulate nIsochromats magnetization vectors per component. Their frequency distribution is Lorenzian if component has a T2* value, otherwise uniform.

    Args:
        config: configuration dictionary.
        component:  component specification from config.
        Meq:    equilibrium magnetization along z axis.
        M0:     initial state of magnetization vector, numpy array of size 3.
        pos:   position (x,y,z) of magnetization vector [m].
        
    Returns:
        component magnetization vectors over time, numpy array of size [nIsochromats, 6, nFrames].  1:3 are magnetization, 4:6 are position.

    '''
    if not M0:
        M0 = [0, 0, Meq] # Default initial state is equilibrium magnetization
    if not pos:
        pos = [0, 0, 0]
    v = [component['vx'], component['vy'], component['vz']]
    D = [component['Dx'], component['Dy'], component['Dz']]
    # Shifts in ppm for dephasing vectors:
    isochromats = [(2*i+1-config['nIsochromats'])/2*config['isochromatStep']+component['CS'] for i in range(0, config['nIsochromats'])]
    comp = np.empty((config['nIsochromats'],6,len(config['t'])))

    for m, isochromat in enumerate(isochromats):
        w = config['w0']*isochromat*1e-6  # Demodulated frequency [kRad / s]
        comp[m,:,:] = applyPulseSeq(config, Meq, M0, w, component['T1'], component['T2'], pos, v, D)
        if 'T2*' in component:
            R2prim = 1/component['T2*'] - 1/component['T2']
            comp[m,:,:] /= 1 + (w/R2prim)**2 # Lorenzian lineshape
    return comp


def getComposants(config, vectors):
    '''Adds transverse and longitudinal composants for each component with composants=True.
    
    Args:
        config: configuration dictionary.
        vectors: magnetization vectors of shape [nx, ny, nz, nComps, nIsochromats, 6, len(config['t'])]

    Returns:
        config: configuration dictionary with added components representing composants.
        vectors: magnetization vectors with added components representing composants.

    '''
    
    for c in range(len(config['components'])):
        if config['components'][c]['composants']:
            Mz = np.expand_dims(vectors[:,:,:,c,:,:,:], axis=3).copy() * (1-1e-3) # Truncate to assert composants are rendered behind
            Mxy = Mz.copy()
            Mz[:,:,:,0,:,:2,:] = 0. # Null transverse magnetization to get longitudinal composant
            Mxy[:,:,:,0,:,2,:] = 0. # Null longitudinal magnetization to get transverse composant
            vectors = np.concatenate((vectors, Mz, Mxy), 3) # Add composants as new components
            
            for subscript, color in [('{z}', [.1, .4, .5]), ('{xy}', [.5, .3, .2])]:
                config['components'].append({'name': '$M_{}$'.format(subscript), 'composants': False})
                colors['comps'].insert(len(config['components'])-1, color)
            
            config['components'][c]['composants'] = False # No need to create more composants

    return config, vectors


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
    ''' Round off time to certain precision.

    Args:
        time: time to be rounded [ms]

    Returns:
        rounded time

    '''

    return np.round(time, decimals=6) # nanosecond precision should be enough


def addEventsToTimeVector(t, pulseSeq):
    ''' Read event times from pulseSeq struct and add to input time vector.

    Args:
        t: input time vector [ms]
        pulseSeq: pulse sequence struct of events with event times [ms]

    Returns:
        Array of unique sorted set of times in input time vector and event times.

    '''

    t = list(t)
    for event in pulseSeq:
        t.append(event['t'])
    return np.unique(roundEventTime(np.array(t)))


def calculateFA(B1, dur):
    ''' Calculate flip angle for given B1 waveform and duration.

    Args:
        B1: vector of B1 amplitudes [uT]
        dur: duration of pulse [ms]

    Returns:
        Pulse flip angle

    '''

    dwell = dur / len(B1)
    FA = 0
    for b in B1:
        FA += 360*(dwell * gyro * np.real(b) * 1e-6)
    return FA


def loadGradfromFile(filename):
    ''' Read gradient waveform from file and return as list. The file is expected to contain a yaml list of the gradient in mT/m, or a field 'grad' holding such a list.

    Args:
        filename: filename of gradient yaml file.

    Returns:
        Gradient waveform as a list

    '''
    with open(filename, 'r') as f:
        try:
            grad = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise Exception('Error reading gradient file {}'.format(filename)) from exc
        if 'grad' in grad:
            grad = grad['grad']
        if isinstance(grad, list) and len(grad)>0 and isinstance(grad[0], Number):
            return grad
        else:
            raise Exception('Error reading gradient file {}. File must contain a yaml list of numbers.'.format(filename))


def isNumList(obj):
    ''' Check if object is non-empty list of numbers.

    Args:
        obj: object to be checked

    Returns:
        true or false

    '''
    return isinstance(obj, list) and len(obj)>0 and isinstance(obj[0], Number)


def RFfromStruct(RF):
    ''' Read RF pulse from struct and return as array.

    Args:
        RF: list of the RF amplitude, or a struct with key 'amp' and optionally 'phase', each containing a list of equal length. amp is the RF amplitude [uT], and 'phase' is RF phase modulation [degrees].

    Returns:
        RF pulse as a (possibly complex) numpy array

    '''

    if isNumList(RF):
        B1 = np.array(RF)
    elif 'amp' in RF and isNumList(RF['amp']):
        B1 = np.array(RF['amp'])
        if 'phase' in RF:
            if not isNumList(RF['phase']):
                raise Exception("'phase' of RF struct must be numerical list")
            elif len(RF['phase']) != len(B1):
                raise Exception("'amp' and 'phase' of RF struct must have equal length")
            B1 = B1 * np.exp(1j*np.radians(RF['phase']))
    else: 
        raise Exception('Unknown format of RF struct')
    return B1


def loadRFfromFile(filename):
    ''' Read RF pulse from file and return as array. The file is expected to contain a yaml list of the RF amplitude, or a list containing two lists, where the second holds the RF phase in degrees.

    Args:
        filename: filename of RF yaml file.

    Returns:
        RF pulse as a numpy array (complex if phase was given)

    '''
    with open(filename, 'r') as f:
        try:
            RFstruct = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise Exception('Error reading RF file {}'.format(filename)) from exc
    return RFfromStruct(RFstruct)


def checkPulseSeq(config):
    ''' Check/verify pulse sequence given by config.
    
    Args:
        config: configuration dictionary.
    '''

    if 'pulseSeq' not in config:
        config['pulseSeq'] = []
    allowedKeys = ['t', 'spoil', 'dur', 'FA', 'B1', 'phase', 'Gx', 'Gy', 'Gz']
    for event in config['pulseSeq']:
        for item in event.keys(): # allowed keys
            if item not in allowedKeys:
                raise Exception('PulseSeq key "{}" not supported'.format(item))
        if not 't' in event:
            raise Exception('All pulseSeq events must have an event time "t"')
        if not any([key in event for key in ['FA', 'B1', 'Gx', 'Gy', 'Gz', 'spoil']]):
            raise Exception('Empty events not allowed')
        if event['t'] > config['TR']:
            raise Exception('pulseSeq event t exceeds TR')
        if 'spoil' in event: # Spoiler event
            if not event['spoil']:
                raise Exception('Spoiler event must have spoil: true')
            if any([key not in ['t', 'spoil'] for key in event]):
                raise Exception('Spoiler event should only have event time t and spoil: true')
            event['spoiltext'] = 'spoiler'
        else:
            if 'dur' not in event:
                raise Exception('All pulseSeq events except spoiler events must have a duration dur [msec]')
            if roundEventTime(event['dur'])==0:
                raise Exception('Event duration is too short')
            if (event['t'] + event['dur']) > config['TR']:
                raise Exception('pulseSeq event t+dur exceeds TR')
        if 'phase' in event:
            if not isinstance(event['phase'], Number):
                raise Exception('Event phase [degrees] must be numeric')
            if not ('FA' in event or 'B1' in event):
                raise Exception('Only RF events can have a phase')

        if 'FA' in event or 'B1' in event: # RF-pulse event (possibly with gradient)

            # combinations not allowed:
            if 'B1' in event and not 'dur' in event:
                raise Exception('RF-pulse must provide "dur" along with "B1"')
            
            if 'B1' in event:
                if isinstance(event['B1'], Number):
                    event['B1'] = np.array([event['B1']])
                elif isinstance(event['B1'], str):
                    event['B1'] = loadRFfromFile(event['B1'])
                else:
                    event['B1'] = RFfromStruct(event['B1'])
            
            # calculate FA prescribed by B1
            if 'B1' in event:
                calcFA = calculateFA(event['B1'], event['dur'])

            # calculate B1 or scale it to get prescribed FA
            if 'FA' in event:
                if 'B1' not in event:
                    event['B1'] = np.array([event['FA']/(event['dur'] * 360 * gyro * 1e-6)])
                else:
                    event['B1'] = event['B1'] * event['FA'] / calcFA # scale B1 to get prescribed FA
            else:
                event['FA'] = calcFA

            event['w1'] = [2 * np.pi * gyro * B1 * 1e-6 for B1 in event['B1']] # kRad / s
            event['RFtext'] = str(int(abs(event['FA'])))+u'\N{DEGREE SIGN}'+'-pulse'
        if any([key in event for key in ['Gx', 'Gy', 'Gz']]): # Gradient (no RF)
            if not ('dur' in event and event['dur']>0):
                raise Exception('Gradient must have a specified duration>0 (dur [ms])')
            for g in ['Gx', 'Gy', 'Gz']:
                if g in event:
                    if isinstance(event[g], dict) and 'file' in event[g] and 'amp' in event[g]:
                        grad = loadGradfromFile(event[g]['file'])
                        event[g] = list(np.array(grad) / np.max(grad) * event[g]['amp'])
                    elif not isinstance(event[g], Number) and not (isinstance(event[g], list) and len(event[g])>0):
                        raise Exception('Unknown type {} for B1'.format(type(event[g])))
    
    # Sort pulseSeq according to event time
    config['pulseSeq'] = sorted(config['pulseSeq'], key=lambda event: event['t'])

    # split any pulseSeq events with array values into separate events
    config['separatedPulseSeq'] = []
    for event in config['pulseSeq']:
        arrLengths = [len(event[key]) for key in ['w1', 'Gx', 'Gy', 'Gz'] if key in event and isinstance(event[key], list)]
        if len(arrLengths) > 0: # arrays in event
            arrLength = np.max(arrLengths)
            if len(set(arrLengths))==2 and 1 in set(arrLengths):
                # extend any singleton arrays to full length
                for key in ['w1', 'Gx', 'Gy', 'Gz']:
                    if key in event and isinstance(event[key], list) and len(event[key])==1:
                        event[key] *= arrLength
            elif len(set(arrLengths))>1:
                raise Exception('If w1, Gx, Gy, Gz of an event are provided as lists, equal length is required')
            for i, t in enumerate(np.linspace(event['t'], event['t'] + event['dur'], arrLength, endpoint=False)):
                subDur = event['dur'] / arrLength
                subEvent = {'t': t, 'dur': subDur}
                if i==0 and spoil in event:
                    subEvent['spoil'] = event['spoil']
                for key in ['w1', 'Gx', 'Gy', 'Gz', 'phase', 'RFtext']:
                    if key in event:
                        if type(event[key]) is list:
                            if i < len(event[key]):
                                subEvent[key] = event[key][i]
                            else:
                                raise Exception('Length of {} does not match other event properties'.format(key))
                        else:
                            subEvent[key] = event[key]
                        if key in ['Gx', 'Gy', 'Gz']:
                            subEvent['{}text'.format(key)] = '{}: {:2.0f} mT/m'.format(key, subEvent[key])
                config['separatedPulseSeq'].append(subEvent)
        else:
            for key in ['Gx', 'Gy', 'Gz']:
                if key in event:
                    event['{}text'.format(key)] = '{}: {:2.0f} mT/m'.format(key, event[key])
            config['separatedPulseSeq'].append(event)

    # Sort separatedPulseSeq according to event time
    config['separatedPulseSeq'] = sorted(config['separatedPulseSeq'], key=lambda event: event['t'])


def emptyEvent():
    ''' Creates empty pulse sequence event.

    Returns:
        "empty" pulse sequence event

    '''

    return {'w1': 0, 'Gx': 0, 'Gy': 0, 'Gz': 0, 'phase': 0, 'spoil': False}


def mergeEvent(event, event2merge, t):
    ''' Merge events by adding w1, Gx, Gy, Gz, phase and updating event texts. Also update event time t.

    Args:
        event: original event
        event2merge: event to be merged
        t:  new event time

    Returns:
        Merged event

    '''

    for channel in ['w1', 'Gx', 'Gy', 'Gz', 'phase']:
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
    ''' Detach events by subtracting w1, Gx, Gy, Gz, phase and removing event texts. Also update event time t.

    Args:
        event: original event
        event2detach: event to be detached
        t:  new event time

    Returns:
        Detached event

    '''

    for channel in ['w1', 'Gx', 'Gy', 'Gz', 'phase']:
        if channel in event2detach:
            event[channel] -= event2detach[channel]
    for text in ['RFtext', 'Gxtext', 'Gytext', 'Gztext', 'spoilText']:
        if text in event and text in event2detach and event[text]==event2detach[text]:
            del event[text]
    event['t'] = t
    return event


def getPrescribedTimeVector(config, nTR):
    ''' Get time vector of animations prescribed by 'speed', 'TR', 'fps', and 'maxRFspeed' in config.

    Args:
        config: configuration dictionary
        nTR:    number of TR:s in time vector

    Returns:
        Time vector prescribed by config

    '''

    speedEvents = config['speed'] + [event for event in config['pulseSeq'] if any(['FA' in event, 'B1' in event])]
    speedEvents = sorted(speedEvents, key=lambda event: event['t'])
    
    kernelTime = np.array([])
    t = 0
    dt = 1e3 / config['fps'] * config['speed'][0]['speed'] # Animation time resolution [msec]
    for event in speedEvents:
        kernelTime = np.concatenate((kernelTime, np.arange(t, event['t'], dt)), axis=None)
        t = max(t, event['t'])
        if 'speed' in event:
            dt = 1e3 / config['fps'] * event['speed'] # Animation time resolution [msec]
        if 'FA' in event or 'B1' in event:
            RFdt = min(dt, 1e3 / config['fps'] * config['maxRFspeed']) # Time resolution during RF [msec]
            kernelTime = np.concatenate((kernelTime, np.arange(event['t'], event['t'] + event['dur'], RFdt)), axis=None)
            t = event['t'] + event['dur']
    kernelTime = np.concatenate((kernelTime, np.arange(t, config['TR'], dt)), axis=None)

    timeVec = np.array([])
    for rep in range(nTR): # Repeat time vector for each TR
        timeVec = np.concatenate((timeVec, kernelTime + rep * config['TR']), axis=None)
    return np.unique(roundEventTime(timeVec))


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
        eventTime = roundEventTime(event['t'])
        # Merge any events starting simultaneously:
        if eventTime==newEvent['t']:
            newEvent = mergeEvent(newEvent, event, eventTime)
        else:
            config['events'].append(dict(newEvent))
            newEvent = mergeEvent(newEvent, event, eventTime)
        if 'dur' in event: # event is ongoing unless no 'dur', i.e. spoiler event
            ongoingEvents.append(event)
            # sort ongoing events according to event end time:
            sorted(ongoingEvents, key=lambda event: event['t'] + event['dur'], reverse=False)
        if event is config['separatedPulseSeq'][-1]:
            nextEventTime = roundEventTime(config['TR'])
        else:
            nextEventTime = roundEventTime(config['separatedPulseSeq'][i+1]['t'])
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
    for rep in range(-config['nDummies'], config['nTR']): # Repeat time vector for each TR (dummy TR:s get negative time)
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
    if 'nDummies' not in config:
        config['nDummies'] = 0
    if 'nIsochromats' not in config:
        config['nIsochromats'] = 1
    if 'isochromatStep' not in config:
        if config['nIsochromats']>1:
            raise Exception('Please specify "isochromatStep" [ppm] in config')
        else:
            config['isochromatStep']=0
    if 'plotB1' not in config:
        config['plotB1'] = False
    if 'components' not in config:
        config['components'] = [{}]
    for c, comp in enumerate(config['components']):
        for (key, default) in [('name', ''), 
                               ('CS', 0), 
                               ('T1', np.inf), 
                               ('T2', np.inf), 
                               ('vx', 0), 
                               ('vy', 0), 
                               ('vz', 0), 
                               ('Dx', 0), 
                               ('Dy', 0), 
                               ('Dz', 0),
                               ('composants', False)]:
            if key not in comp:
                comp[key] = default
        if 'color' in comp:
            colors['comps'].insert(c, comp['color'])
    config['nComps'] = len(config['components'])
    if 'locSpacing' not in config:
        config['locSpacing'] = 0.001      # distance between locations [m]

    if 'fps' not in config:
        config['fps'] = 15 # Frames per second in animation (<=15 should be supported by powepoint)
    
    config['w0'] = 2*np.pi*gyro*config['B0'] # Larmor frequency [kRad/s]

    # check speed prescription
    if isinstance(config['speed'], Number):
        config['speed'] = [{'t': 0, 'speed': config['speed']}]
    elif isinstance(config['speed'], list):
        for event in config['speed']:
            if not ('t' in event and 'speed' in event):
                raise Exception("Each item in 'speed' list must have field 't' [msec] and 'speed'")
            if event['t']>=config['TR']:
                raise Exception("Specified speed change must be within TR.")
        if not 0 in [event['t'] for event in config['speed']]:
            raise Exception("Speed at time 0 must be specified.")
    else:
        raise Exception("Config 'speed' must be a number or a list")
    config['speed'] = sorted(config['speed'], key=lambda event: event['t'])
    if 'maxRFspeed' not in config:
        config['maxRFspeed'] = 0.001
    elif not isinstance(config['maxRFspeed'], Number):
        raise Exception("Config 'maxRFspeed' must be numeric")

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
        if 'tRange' in output:
            if not len(output['tRange'])==2:
                raise Exception('Output "tRange" expected to be 2-tuple')
        elif output['type']=='psd':
            output['tRange'] = [0, config['TR']]
        else:
            output['tRange'] = [0, config['nTR'] * config['TR']]
        if 'dpi' not in output:
            output['dpi'] = 100
        if 'freeze' not in output:
            output['freeze'] = []
        elif not isinstance(output['freeze'], list):
            output['freeze'] = [output['freeze']]
        if output['type']=='3D':
            if 'drawAxes' not in output:
                output['drawAxes'] = config['nx']*config['ny']*config['nz'] == 1
            if 'azimuth' not in output:
                output['azimuth'] = -78 # azimuthal angle [deg] of x-y-plane
            if 'elevation' not in output:
                output['elevation'] = None # elevation angle  [deg]
            if 'legend' not in output:
                output['legend'] = True # plot figure legend
        elif output['type']=='xy':
            if 'sum' not in output:
                output['sum'] = False # sum transverse components
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


def resampleOnPrescribedTimeFrames(vectors, B1vector, config):
    ''' Resample (interpolate) given vectors corresponding to time vector config['t'] on time vector config['tFrames]. Also resample text and alpha channels in config similiarly.

    Args:
        vectors:    magnetization vectors of shape [nx, ny, nz, nComps, nIsochromats, 6, len(config['t'])]
        B1vector:   complex numpy array of size [nFrames].

    Returns:
        resampled magnetization vectors of shape [nx, ny, nz, nComps, nIsochromats, 6, len(config['tFrames'])]

    '''

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
    B1vector = np.interp(config['tFrames'], config['t'], B1vector)

    # resample text alpha channels:
    for channel in ['RFalpha', 'Galpha', 'spoilAlpha']:
        alphaVector = np.zeros([len(config['tFrames'])])
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

    return resampledVectors, B1vector


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


def run(configFile, leapFactor=1):
    ''' Main program. Read and setup config, simulate magnetization vectors and write animated gif.
        
    Args:
        configFile: YAML file specifying configuration.
        leapFactor: Skip frame factor for faster processing and smaller filesize.
        
    '''

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

    if config['background']['color'] == 'black' and colors['bg'][0] == 1:
        for i in ['bg', 'axis', 'text', 'circle']:
            colors[i][:3] = list(map(lambda x: 1-x, colors[i][:3]))

    ### Simulate ###
    vectors = np.empty((config['nx'],config['ny'],config['nz'],config['nComps'],config['nIsochromats'],6,len(config['t'])))
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
                    pos = [(x+.5-config['nx']/2)*config['locSpacing'],
                           (y+.5-config['ny']/2)*config['locSpacing'],
                           (z+.5-config['nz']/2)*config['locSpacing']]
                    vectors[x,y,z,c,:,:,:] = simulateComponent(config, component, Meq, M0, pos)
    
    B1vector = getB1vector(config)

    ### Animate ###
    getText(config) # prepare text flashes for 3D plot
    vectors, B1vector = resampleOnPrescribedTimeFrames(vectors, B1vector, config)
    fadeTextFlashes(config)

    outPath = Path(__file__).parent.parent / 'out'
    for output in config['output']:
        if output['file']:
            if output['type'] in ['xy', 'z']:
                signal = np.sum(vectors[:,:,:,:,:,:3,:], (0,1,2,4)) # sum over space and isochromats
                if 'normalize' in output and output['normalize']:
                    for c, comp in enumerate([n['name'] for n in config['components']]):
                        if comp in config['locations']:
                            signal[c,:] /= np.sum(config['locations'][comp])
                signal /= np.max(np.abs(signal)) # scale signal relative to maximum
                if 'scale' in output:
                    signal *= output['scale']
            if output['type']=='3D':
                if any(comp['composants'] for comp in config['components']):
                    config, vectors = getComposants(config, vectors)
            ffmpegWriter = FFMPEGwriter.FFMPEGwriter(config['fps'])
            outPath.mkdir(exist_ok=True)
            outFile = outPath / output['file']
            
            output['freezeFrames'] = []
            for t in output['freeze']:
                output['freezeFrames'].append(np.argmin(np.abs(config['tFrames'] - t)))
            for frame in range(0, len(config['tFrames']), leapFactor):
                # Use only every leapFactor frame in animation
                if output['type'] == '3D':
                    fig = plotFrame3D(config, vectors, B1vector, frame, output)
                elif output['type'] == 'kspace':
                    fig = plotFrameKspace(config, frame, output)
                elif output['type'] == 'psd':
                    fig = plotFramePSD(config, frame, output)
                elif output['type'] in ['xy', 'z']:
                    fig = plotFrameMT(config, signal, frame, output)
                plt.draw()

                filesToSave = []
                if frame in output['freezeFrames']:
                    filesToSave.append(outFile.parent / (outFile.stem + '_{}.png'.format(str(frame).zfill(4))))

                ffmpegWriter.addFrame(fig)
                
                for file in filesToSave:
                    print('Saving frame {}/{} as "{}"'.format(frame+1, len(config['tFrames']), file))
                    plt.savefig(file, facecolor=plt.gcf().get_facecolor())

                plt.close()
            ffmpegWriter.write(outFile)


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
