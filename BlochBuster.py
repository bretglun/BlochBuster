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
from matplotlib.patches import Circle
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
            'spoilText':[128/256,0,0],
            'RFtext':   [0,128/256,0],
            'Gtext':    [80/256,80/256,0],
            'comps': [  [.3,.5,.2],
                        [.1,.4,.5],
                        [.5,.3,.2],
                        [.5,.4,.1],
                        [.4,.1,.5],
                        [.6,.1,.3]]}

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
def plotFrame3D(names, locs, title, clock, frame, spoilAlpha, Galpha, RFalpha, Gtext, RFtext):
    nx, ny = locs.shape
    nz = 1
    xpos = np.arange(nx)-nx/2+.5
    ypos = -(np.arange(ny)-ny/2+.5)

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
    fig.text(.5, 1, title, fontsize=14, horizontalalignment='center', verticalalignment='top', color=colors['text'])

    # Draw time
    time_text = fig.text(0, 0, 'time = %.1f msec' % (clock[frame%len(clock)]), color=colors['text'], verticalalignment='bottom')

    # Draw magnetization vectors
    for y in range(ny):
        for x in range(nx):
            comps = locs[x,y]
            nVecs = len(comps[0])
            order = [int((nVecs-1)/2-abs(m-(nVecs-1)/2)) for m in range(nVecs)]
            for c in range(len(comps)):
                for m in range(nVecs):
                    col = colors['comps'][(c) % len(colors['comps'])]
                    M = comps[c][m][:,frame]
                    Mnorm = np.linalg.norm(M)
                    alpha = 1.-2*np.abs((m+.5)/nVecs-.5)
                    '''
                    if m == nVecs//2:  # Just for getting labels
                        ax.plot([0, 0], [0, 0], [0, 0], '-', lw=2, color=col, alpha=1.,
                                label=names[c])
                                '''
                        # TODO: fix legend for multiple locations
                    if Mnorm>.05:
                        ax.add_artist(Arrow3D([xpos[x], xpos[x]+M[0]], [ypos[y], ypos[y]+M[1]],
                                            [0, M[2]], mutation_scale=20,
                                            arrowstyle="-|>", lw=2,
                                            color=col, alpha=alpha, 
                                            zorder=order[m]+nVecs*int(100*(1-Mnorm))))
    
    # Draw "spoiler" and "FA-pulse" text
    fig.text(1, .94, RFtext[frame], fontsize=14, alpha=RFalpha[frame],
            color=colors['RFtext'], horizontalalignment='right', verticalalignment='top')
    fig.text(1, .88, Gtext[frame], fontsize=14, alpha=Galpha[frame],
            color=colors['Gtext'], horizontalalignment='right', verticalalignment='top')
    fig.text(1, .82, 'spoiler', fontsize=14, alpha=spoilAlpha[frame],
            color=colors['spoilText'], horizontalalignment='right', verticalalignment='top')
    
    # Draw legend:
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
                    [plt.Line2D((0, 1), (0, 0), lw=2, color=colors['comps'][(c) %
                                len(colors['comps'])]) for c, handle in enumerate(
                                handles)], labels, loc=2, bbox_to_anchor=[
                                .14, .83])
    leg.draw_frame(False)
    for text in leg.get_texts():
        text.set_color(colors['text'])
    
    return fig


# Creates an animated plot of magnetization over time plotType='xy' for transversal and 'z' for longitudinal
def plotFrameMT(names, locs, title, clock, frame, plotType):
    if plotType not in ['xy', 'z']:
        raise Exception(
             'plotType must be xy (for transversal) or z (for longitudinal)')

    # create diagram
    xmin, xmax = 0, clock[-1]
    if plotType == 'xy':
        ymin, ymax = 0, 1
    elif plotType == 'z':
        ymin, ymax = -1, 1
    fig = plt.figure(figsize=(5, 2.7), facecolor=colors['bg'])
    ax = fig.gca(xlim=(xmin, xmax), ylim=(ymin, ymax), fc=colors['bg'])
    for side in ['bottom', 'right', 'top', 'left']:
        ax.spines[side].set_visible(False)  # remove default axes
    ax.grid()
    plt.title(title, color=colors['text'])
    plt.xlabel('time[ms]', horizontalalignment='right', color=colors['text'])
    if plotType == 'xy':
        ax.xaxis.set_label_coords(1.1, .1)
        plt.ylabel('$|M_{xy}|$', rotation=0, color=colors['text'])
    elif plotType == 'z':
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
    comps = locs[0,0] #TODO: sum magnetization from all locations
    nVecs = len(comps[0])
    if plotType == 'xy':
        Msum = np.zeros([2, frame+1])
        for c in range(len(comps)):
            col = colors['comps'][(c) % len(colors['comps'])]
            Mxy = np.zeros([2, frame+1])
            for m in range(nVecs):
                Mxy += comps[c][m][:2, :frame+1]
            ax.plot(clock[:frame+1], np.linalg.norm(Mxy, axis=0)/nVecs, '-', lw=2, color=col, label=names[c])
            Msum += Mxy
        # Special case: also plot sum of fat and water
        if 'water' in names and 'fat' in names:
            col = colors['comps'][(len(comps)) % len(colors['comps'])]
            ax.plot(clock[:frame+1], np.linalg.norm(Msum, axis=0)/nVecs/len(comps), '-', lw=2, color=col, label=names[c])
    elif plotType == 'z':
        for c in range(len(comps)):
            col = colors['comps'][(c) % len(colors['comps'])]
            Mz = np.zeros([frame+1])
            for m in range(nVecs):
                Mz += comps[c][m][2, :frame+1]
            ax.plot(clock[:frame+1], Mz/nVecs, '-', lw=2, color=col, label=names[c])

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
def applyPulseSeq(Meq, w, T1, T2, pulseSeq, clock, TR, nTR=1, xpos=0, ypos=0):
    # Initial state is equilibrium magnetization
    M = np.array([[0.], [0.], [Meq]])
    for rep in range(nTR):
        lastFrame = 0
        for event in pulseSeq:
            # Relaxation up to event
            T = clock[event['frame']]-clock[lastFrame]
            t = np.linspace(0, T, event['frame']-lastFrame+1, endpoint=True)
            M1 = integrate.odeint(derivs, M[:, -1], t, args=(Meq, w, 0., T1, T2))
            M = np.concatenate((M, M1[1:].transpose()), axis=1)

            wg = w  # frequency due to w plus any gradients
            if 'w1' in event:
                w1 = event['w1']
            else:
                w1 = 0
            t = np.linspace(0, event['nFrames']*dt, event['nFrames']+1, endpoint=True)

            if 'spoil' in event and event['spoil']: # Spoiler event
                M[:, -1] = spoil(M[:, -1])
            else:
                if 'FA' in event and event['dur']==0: # "instant" RF-pulse event (incompatible with gradient)
                    M1 = integrate.odeint(derivs, M[:, -1], t, args=(Meq, 0., w1, np.inf, np.inf))
                else: # RF-pulse and/or gradient event
                    if any(key in event for key in ['Gx', 'Gy']): # Gradient present
                        if 'Gx' in event:
                            wg += 2*np.pi*gyro*event['Gx']/1000*(xpos*locSpacing) # [krad/s]
                        if 'Gy' in event:
                            wg += 2*np.pi*gyro*event['Gy']/1000*(ypos*locSpacing) # [krad/s]
                    M1 = integrate.odeint(derivs, M[:, -1], t, args=(Meq, wg, w1, T1, T2))
                M = np.concatenate((M, M1[1:].transpose()), axis=1)
    
            lastFrame = event['frame']+event['nFrames']

        # Then relaxation until end of TR
        T = clock[-1]-clock[lastFrame]
        t = np.linspace(0, T, len(clock)-lastFrame, endpoint=True)
        M1 = integrate.odeint(derivs, M[:, -1], t, args=(Meq, w, 0., T1, T2))
        M = np.concatenate((M, M1[1:].transpose()), axis=1)
    return M


# Simulate Nisochromats dephasing magnetization vectors per component
def simulateComponent(component, Meq, w0, Nisochromats, isochromatStep, pulseSeq, clock, TR, nTR=1, xpos=0, ypos=0):
    # Shifts in ppm for dephasing vectors:
    isochromats = [(2*i+1-Nisochromats)/2*isochromatStep+component['CS'] for i in range(0, Nisochromats)]
    comp = []
    for isochromat in isochromats:
        w = w0*isochromat*1e-6  # Demodulated frequency [krad]
        comp.append(applyPulseSeq(Meq, w, component['T1'], component['T2'], pulseSeq, clock, TR, nTR, xpos, ypos))
    return comp


# Get clock during nTR applications of pulseSeq (clock stands still during excitation)
# Get opacity and text for spoiler and RF text flashes in 3D plot
def getText(pulseSeq, TR, nTR, clock):
    framesSinceSpoil = [np.inf]
    framesSinceG = [np.inf]
    framesSinceRF = [np.inf]
    Gtext = ['']
    RFtext = ['']
    for rep in range(nTR):
        lastFrame = 0
        for event in pulseSeq:
            # Frames during relaxation
            nFrames = event['frame']-lastFrame
            count = np.linspace(0, nFrames, nFrames+1, endpoint=True)
            framesSinceSpoil.extend(framesSinceSpoil[-1] + count[1:])
            framesSinceG.extend(framesSinceG[-1] + count[1:])
            framesSinceRF.extend(framesSinceRF[-1] + count[1:])
            RFtext += [RFtext[-1]]*nFrames
            Gtext += [Gtext[-1]]*nFrames
            
            #Frames during event
            count = np.linspace(0, event['nFrames'], event['nFrames']+1, endpoint=True)
            
            if 'spoil' in event and event['spoil']: # Spoiler event
                framesSinceSpoil[-1] = 0
            framesSinceSpoil.extend(framesSinceSpoil[-1] + count[1:])
            if 'FA' in event: # RF-pulse and/or gradient event
                framesSinceRF[-1] = 0
                framesSinceRF.extend([0]*event['nFrames'])
                # TODO: add info about the RF phase angle
                RFtext[-1] = str(int(abs(event['FA'])))+u'\N{DEGREE SIGN}'+'-pulse'
            else:
                framesSinceRF.extend(framesSinceRF[-1] + count[1:])
            RFtext += [RFtext[-1]]*event['nFrames']
            if any(key in event for key in ['Gx', 'Gy']): # gradient event
                framesSinceG[-1] = 0
                framesSinceG.extend([0]*event['nFrames'])
                # TODO: gradient text
                Gtext[-1] = 'gradient'
            else:
                framesSinceG.extend(framesSinceG[-1] + count[1:])
            Gtext += [Gtext[-1]]*event['nFrames']

            lastFrame = event['frame'] + event['nFrames']

        # Frames during relaxation until end of TR
        nFrames = len(clock)-lastFrame
        count = np.linspace(0, nFrames, nFrames+1, endpoint=True)
        framesSinceSpoil.extend(framesSinceSpoil[-1] + count[1:])
        framesSinceG.extend(framesSinceG[-1] + count[1:])
        framesSinceRF.extend(framesSinceRF[-1] + count[1:])
        RFtext += [RFtext[-1]]*nFrames
        Gtext += [Gtext[-1]]*nFrames
            
    # Calculate alphas
    decrPerFrame = .1 #TODO: include fps
    spoilAlpha = [max(1.0-frames*decrPerFrame, 0) for frames in framesSinceSpoil]
    Galpha = [max(1.0-frames*decrPerFrame, 0) for frames in framesSinceG]
    RFalpha = [max(1.0-frames*decrPerFrame, 0) for frames in framesSinceRF]
    
    return spoilAlpha, Galpha, RFalpha, Gtext, RFtext


# TODO: nicer without function and zfill
def filename(dir, frame): return dir + '/' + format(frame+1, '04') + '.png'


def checkPulseSeq(config):
    allowedKeys = ['t', 'spoil', 'dur', 'FA', 'B1', 'phase', 'Gx', 'Gy']
    for event in config['pulseSeq']:
        for item in event.keys(): # allowed keys
            if item not in allowedKeys:
                raise Exception('PulseSeq key "{}" not supported'.format(item))
        if not 't' in event:
            raise Exception('All pulseSeq events must have an event time "t"')    

    # Sort pulseSeq according to event time
    config['pulseSeq'] = sorted(config['pulseSeq'], key=lambda event: event['t'])
    
    t = np.array([0.0])
    for event in config['pulseSeq']:
        T = np.round(event['t']/dt)*dt-t[-1] # time up to event
        if T<0:
            raise Exception('Pulse sequence events overlap')
        t = np.append(t[:-1], t[-1]+np.linspace(0, T, np.round(T/dt)+1, endpoint=True))
        event['frame'] = len(t)-1 # starting frame of event TODO: can be removed?
            
        if 'spoil' in event: # Spoiler event
            if any(key in event for key in ['dur', 'FA', 'B1', 'phase', 'Gx', 'Gy']):
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
            if 'dur' not in event:
                event['dur'] = abs(event['FA'])/(360*gyro*event['B1']*1e-6) # RF pulse duration
            if 'FA' not in event:
                event['FA'] = 360*(event['dur']*gyro*event['B1']*1e-6) # calculate prescribed FA
            if event['dur']>0:
                event['nFrames'] = int(max(np.round(event['dur']/dt), 1))
            else:
                event['nFrames'] = int(np.round(abs(event['FA'])*18/90)) # 18 frames per 90 flip TODO: use fps
            if 'phase' in event: # Set complex flip angles
                event['FA'] = event['FA']*np.exp(1j*radians(event['phase']))
            event['w1'] = radians(event['FA'])/(event['nFrames']*dt)

        if 'Gx' in event or 'Gy' in event: # Gradient (no RF)
            if not ('dur' in event and event['dur']>0):
                raise Exception('Gradient must have a specified duration>0 (dur [ms])')
            if 'FA' not in event and 'phase' in event:
                raise Exception('Gradient event should have no phase')
            event['nFrames'] = int(max(np.round(event['dur']/dt), 1))
            if 'Gx' in event:
                Gx = event['Gx']*event['nFrames']*dt/event['dur'] # adjust Gx for truncation of duration
            if 'Gy' in event:
                Gy = event['Gy']*event['nFrames']*dt/event['dur'] # adjust Gy for truncation of duration
            
        if event['dur']>0:
            event['dur'] = event['nFrames']*dt  # truncate duration to whole frames
        t = np.append(t[:-1], t[-1]+np.linspace(0, event['dur'], event['nFrames']+1, endpoint=True))
        
    T = np.ceil(config['TR']/dt)*dt-t[-1] # time up to TR
    if np.round(T/dt)<0:
        raise Exception('Pulse sequence events not within TR')
    t = np.append(t[:-1], t[-1]+np.linspace(0, T, np.round(T/dt)+1, endpoint=True))
    return t


# Main program
def BlochBuster(configFile, leapFactor=1, blackBackground=False, useFFMPEG = True):
    # Set global constants
    global gyro, dt, locSpacing
    
    if blackBackground:
        for i in ['bg', 'axis', 'text', 'circle']:
            colors[i][:3] = list(map(lambda x: 1-x, colors[i][:3]))
    # Read configuration file
    with open(configFile, 'r') as f:
        try:
            config = yaml.load(f)
        except yaml.YAMLError as exc:
            print(exc)
    
    ### Calculations ###
    gyro = 42577.			# Gyromagnetic ratio [kHz/T]
    fps = 15.				# Frames per second in animation (<=15 should be supported by powepoint)
    dt = 1e3/fps*config['speed'] 	# Time resolution [msec]
    w0 = 2*np.pi*gyro*config['B0']  # Larmor frequency [kRad/s]
    
    ### Format pulseSeq correctly ###
    clock = checkPulseSeq(config)
    
    ### Arrange locations ###
    locSpacing = 0.001      # distance between locations [m]
    if not 'locations' in config:
        config['locations'] = [[1]]
        nx = ny = 1
    else:
        if isinstance(config['locations'], dict):
            rows = next(iter(config['locations'].values()))
        elif isinstance(config['locations'], list):
            rows = config['locations']
        else:
            raise Exception('Did not expect {} in config "locations"'.format(type(config['locations'])))
        if not isinstance(rows[0], list):
            raise Exception('Config "locations" should be a list of lists, such as [[1, 1]]')
        ny = len(rows)
        nx = len(rows[0])

    ### Simulate ###
    locs = np.empty((nx,ny), dtype=list)
    for y in range(ny):
        for x in range(nx):
            comps = []
            for c, component in enumerate(config['components']):
                if component['name'] in config['locations']:
                    Meq = config['locations'][component['name']][y][x]
                elif isinstance(config['locations'], list):
                    Meq = config['locations'][y][x]
                else:
                    continue
                xpos = x+.5-nx/2
                ypos = y+.5-ny/2
                # TODO: put clock on config and send entire config
                comps.append(simulateComponent(component, Meq, w0, config['nIsochromats'], config['isochromatStep'], config['pulseSeq'], clock, config['TR'], config['nTR'], xpos, ypos))
            locs[x,y] = comps

    ### Animate ###
    spoilAlpha, Galpha, RFalpha, Gtext, RFtext = getText(config['pulseSeq'], config['TR'], config['nTR'], clock)
    delay = int(100/fps*leapFactor)  # Delay between frames in ticks of 1/100 sec
    nFrames = len(locs[0,0][0][0][0])
    if not config['outFile3D']+config['outFileMxy']+config['outFileMz']:
        raise Exception('No outfile (outFile3D/outFileMxy/outFileMz) was found in config')
    tmpdir = './tmp'
    outdir = './out'
    if os.path.isdir(tmpdir):
        rmTmpDir = input('Temporary folder "{}" already exists. Delete(Y/N)?'.format(tmpdir))
        if rmTmpDir.upper() == 'Y':
            shutil.rmtree(tmpdir)
        else:
            raise Exception('No files written.')
    for (plotType, outfile) in [('3D', config['outFile3D']), ('xy', config['outFileMxy']), ('z', config['outFileMz'])]:
        if outfile:
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, outfile)
            if useFFMPEG:
                ffmpegWriter = FFMPEGwriter.FFMPEGwriter(fps)
            else:
                os.makedirs(tmpdir, exist_ok=True)
            names = [comp['name'] for comp in config['components']]
            for frame in range(0, nFrames, leapFactor):
                # Use only every leapFactor frame in animation
                if plotType == '3D':
                    fig = plotFrame3D(names, locs, config['title'], clock, frame, spoilAlpha, Galpha, RFalpha, Gtext, RFtext)
                else:
                    fig = plotFrameMT(names, locs, config['title'], clock, frame, plotType)
                plt.draw()
                if useFFMPEG:
                    ffmpegWriter.addFrame(fig)
                else: # use imagemagick: save frames temporarily 
                    file = filename(tmpdir, frame)
                    print('Saving frame {}/{} as "{}"'.format(frame+1, nFrames, file))
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
