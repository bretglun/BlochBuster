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
# import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.patches import FancyArrowPatch
import numpy as np
import scipy.integrate as integrate
import os.path
import shutil
import csv
import optparse
import yaml
import subprocess

colors = {  'bg': [1,1,1], 
            'circle': [0,0,0,.03],
            'axis': [.5,.5,.5],
            'text': [.05,.05,.05], 
            'spoilText': [80/256,0,0],
            'RFText': [0,80/256,0],
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
def plotFrame3D(names, comps, title, clock, frame, spoilTextAlpha, RFTextAlpha, RFText):
    # Create 3D axes
    fig = plt.figure(figsize=(5, 4.7))
    ax = fig.gca(projection='3d', xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), fc=colors['bg'])
    ax.set_axis_off()
    ax.set_position([-0.26, -0.39, 1.6, 1.58])

    # Draw axes circles
    for i in ["x", "y", "z"]:
        circle = Circle((0, 0), 1, fill=True, lw=1, fc=colors['circle'])
        ax.add_patch(circle)
        art3d.pathpatch_2d_to_3d(circle, z=0, zdir=i)

    # Draw x, y, and z axes
    ax.plot([-1, 1], [0, 0], [0, 0], c=colors['axis'], zorder=-1)  # x-axis
    ax.text(1.1, 0, 0, r'$x^\prime$', horizontalalignment='center', color=colors['text'])
    ax.plot([0, 0], [-1, 1], [0, 0], c=colors['axis'], zorder=-1)  # y-axis
    ax.text(0, 1.2, 0, r'$y^\prime$', horizontalalignment='center', color=colors['text'])
    ax.plot([0, 0], [0, 0], [-1, 1], c=colors['axis'], zorder=-1)  # z-axis
    ax.text(0, 0, 1.1, r'$z$', horizontalalignment='center', color=colors['text'])

    # Draw title:
    ax.text(0, 0, 1.4, title, fontsize=14, horizontalalignment='center', color=colors['text'])
    # Draw time
    time_text = ax.text(-1, -.8, -1, 'time = %.1f msec' % (clock[frame]), color=colors['text'])

    # Draw magnetization vectors
    nVecs = len(comps[0])
    for c in range(len(comps)):
        for m in range(nVecs):
            col = colors['comps'][(c) % len(colors['comps'])]
            M = comps[c][m]
            alpha = 1.-2*np.abs((m+.5)/nVecs-.5)
            order = int((nVecs-1)/2-abs(m-(nVecs-1)/2))
            if m == nVecs//2:  # Just for getting labels
                ax.plot([0, 0], [0, 0], [0, 0], '-', lw=2, color=col, alpha=1.,
                        label=names[c])
            ax.add_artist(Arrow3D([0, M[0, frame]], [0, M[1, frame]],
                                  [0, M[2, frame]], mutation_scale=20,
                                  arrowstyle="-|>", lw=2,
                                  color=col, alpha=alpha, zorder=order))

    # Draw "spoiler" and "FA-pulse" text
    ax.text(.7, .7, .8, 'spoiler', fontsize=14, alpha=spoilTextAlpha[frame],
            color=colors['spoilText'], horizontalalignment='right')
    ax.text(.7, .7, .95, RFText[frame], fontsize=14, alpha=RFTextAlpha[frame],
            color=colors['RFText'], horizontalalignment='right')
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
def plotFrameMT(names, comps, title, clock, frame, plotType):
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
def applyPulseSeq(Meq, w, T1, T2, pulseSeq, TR, w1, nTR=1, dt=0.1, instantRF=False):
    # Initial state is equilibrium magnetization
    M = np.array([[0.], [0.], [Meq]])
    for rep in range(nTR):
        # Initial relaxation up to first pulse
        t = np.arange(0, pulseSeq[0]['t']+dt, dt)
        M1 = integrate.odeint(derivs, M[:, -1], t, args=(Meq, w, 0., T1, T2))
        M = np.concatenate((M, M1[1:].transpose()), axis=1)
        for p, pulse in enumerate(pulseSeq):
            if pulse['FA']==0: # Interpreted as spoiling
                dur = 0
                M[:, -1] = spoil(M[:, -1])
            else: # Apply RF-pulse:
                dur = radians(abs(pulse['FA']))/w1  # RF pulse duration
                t = np.arange(0, dur+dt, dt)
                w1_adj = radians(pulse['FA'])/((len(t)-1)*dt)  # adjust w1 to fit FA to integer number of frames
                if instantRF:
                    dur = 0
                    M1 = integrate.odeint(derivs, M[:, -1], t, args=(Meq, 0., w1_adj, np.inf, np.inf))
                else:
                    M1 = integrate.odeint(derivs, M[:, -1], t, args=(Meq, w, w1_adj, T1, T2))
                M = np.concatenate((M, M1[1:].transpose()), axis=1)
            # Then relaxation until next pulse or end of TR
            if pulse is not pulseSeq[-1]:
                t_next = min(TR, pulseSeq[p+1]['t'])
            else:
                t_next = TR
            T = t_next-pulse['t']-dur
            if T>0:
                t = np.arange(0, T+dt, dt)
                M1 = integrate.odeint(derivs, M[:, -1], t, args=(Meq, w, 0., T1, T2))
                M = np.concatenate((M, M1[1:].transpose()), axis=1)
    return M


# Simulate Nisochromats dephasing magnetization vectors per component
def simulateComponent(component, w0, Nisochromats, isochromatStep, pulseSeq, TR, w1, nTR=1, dt=0.1, instantRF=False):
    # Shifts in ppm for dephasing vectors:
    isochromats = [(2*i+1-Nisochromats)/2*isochromatStep+component['CS'] for i in range(0, Nisochromats)]
    comp = []
    for isochromat in isochromats:
        w = w0*isochromat*.000001  # Demodulated frequency [krad]
        comp.append(applyPulseSeq(component['Meq'], w, component['T1'], component['T2'], pulseSeq, TR, w1, nTR, dt, instantRF))
    return comp


# Get clock during nTR applications of pulseSeq (clock stands still during excitation)
# Get opacity and text for spoiler and RF text flashes in 3D plot
def getClockSpoilAndRFText(pulseSeq, TR, nTR, w1, dt, instantRF=False):
    clock = [0.0]
    decrPerFrame = .1
    spoilTextAlpha = [0.]
    RFTextAlpha = [0.]
    RFText = ['']
    for rep in range(nTR):
        for p, pulse in enumerate(pulseSeq):
            if pulse['FA']==0: # Interpreted as spoiling
                dur = 0
                spoilTextAlpha[-1] = 1.
            else: # Frames during RF-pulse:
                # TODO: add info about the RF phase angle
                RF = str(int(abs(pulse['FA'])))+u'\N{DEGREE SIGN}'+'-pulse'
                dur = radians(abs(pulse['FA']))/w1  # RF pulse duration
                t = np.arange(dt, dur+dt, dt)
                if instantRF:
                    dur = 0
                    clock.extend(np.full(t.shape, clock[-1]))  # Clock stands still during instant RF pulse
                else:
                    clock.extend(t+clock[-1])
                spoilTextAlpha.extend(np.linspace(spoilTextAlpha[-1], spoilTextAlpha[-1]-len(t)*decrPerFrame, num=len(t)))
                RFTextAlpha.extend(np.ones(len(t)))
                RFText += [RF]*len(t)
            # Frames during relaxation
            if pulse is not pulseSeq[-1]:
                t_next = min(TR, pulseSeq[p+1]['t'])
            else:
                t_next = TR
            T = t_next-pulse['t']-dur
            if T>0:
                t = np.arange(dt, T+dt, dt)
                clock.extend(t+clock[-1])  # Increment clock during relaxation time T
                spoilTextAlpha.extend(np.linspace(spoilTextAlpha[-1], spoilTextAlpha[-1]-len(t)*decrPerFrame, num=len(t)))
                RFTextAlpha.extend(np.linspace(RFTextAlpha[-1], RFTextAlpha[-1]-len(t)*decrPerFrame, num=len(t)))
                RFText += [RF]*len(t)
    # Clip at zero
    spoilTextAlpha = [max(alpha, 0) for alpha in spoilTextAlpha]
    RFTextAlpha = [max(alpha, 0) for alpha in RFTextAlpha]
    return clock, spoilTextAlpha, RFTextAlpha, RFText


def filename(dir, frame): return dir + '/' + format(frame+1, '04') + '.png'


# Main program
def BlochBuster(configFile, leapFactor=1, blackBackground=False, useffmpeg = True):
    if blackBackground:
        for i in ['bg', 'axis', 'text', 'circle']:
            colors[i][:3] = list(map(lambda x: 1-x, colors[i][:3]))
    # Read configuration file
    with open(configFile, 'r') as f:
        try:
            config = yaml.load(f)
        except yaml.YAMLError as exc:
            print(exc)
    # Assert pulses in pulseSeq are sorted according to time
    config['pulseSeq'] = sorted(config['pulseSeq'], key=lambda pulse: pulse['t']) 
    # Set complex flip angles
    for pulse in config['pulseSeq']:
        if 'phase' in pulse:
            pulse['FA'] = pulse['FA']*np.exp(1j*radians(pulse['phase']))

    instantRF = config['B1'] >= 100	# B1=100 means instant RF pulses
    config['B1'] /= 1e6			# convert uT->T
    # Calculations
    fps = 15.				# Frames per second in animation (<=15 should be supported by powepoint)
    dt = 1000./fps*config['speed'] 	# Time resolution [msec]
    gyro = 42577.			# Gyromagnetic ratio [kHz/T]
    if instantRF:
        config['B1'] = 1/(gyro*dt*72)  # Set duration of a 360-pulse to 72 frames
    w0 = 2*np.pi*gyro*config['B0']  # Larmor frequency [kRad]
    w1 = 2*np.pi*gyro*config['B1']  # B1 rotation frequency [kRad]
    # Simulate
    comps = []
    for component in config['components']:
        comps.append(simulateComponent(component, w0, config['nIsochromats'], config['isochromatStep'], config['pulseSeq'], config['TR'], w1, config['nTR'], dt, instantRF))
    # Animate
    clock, spoilTextAlpha, RFTextAlpha, RFText = getClockSpoilAndRFText(config['pulseSeq'], config['TR'], config['nTR'], w1, dt, instantRF)
    delay = int(100/fps*leapFactor)  # Delay between frames in ticks of 1/100 sec
    nFrames = len(comps[0][0][0])
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
            if not useffmpeg:
                os.makedirs(tmpdir, exist_ok=True)
            names = [comp['name'] for comp in config['components']]
            for frame in range(0, nFrames, leapFactor):
                # Use only every leapFactor frame in animation
                if plotType == '3D':
                    fig = plotFrame3D(names, comps, config['title'], clock, frame, spoilTextAlpha, RFTextAlpha, RFText)
                else:
                    fig = plotFrameMT(names, comps, config['title'], clock, frame, plotType)
                plt.draw()
                if useffmpeg:
                    if frame == 0:
                        canvas_width, canvas_height = fig.canvas.get_width_height()
                        if '.gif' in outfile:
                            paletteCmd = ('ffmpeg', 
                                '-s', '{}x{}'.format(canvas_width, canvas_height), 
                                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-i', '-', 
                                '-filter_complex', 'palettegen=stats_mode=diff', '-y', 'palette.png')
                            paletteProcess = subprocess.Popen(paletteCmd, stdin=subprocess.PIPE)
                            animationCmd = ('ffmpeg', 
                                '-y', # overwrite output file
                                '-r', str(fps), # frame rate
                                '-s', '{}x{}'.format(canvas_width, canvas_height), # size of image string
                                '-pix_fmt', 'rgb24', # format
                                '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
                                '-i', 'palette.png', '-filter_complex', 'paletteuse',
                                '-vframes', str(len(range(0, nFrames, leapFactor))), # number of frames
                                outfile) # file name
                        elif '.mp4' in outfile:
                            animationCmd = ('ffmpeg', 
                                '-y', # overwrite output file
                                '-r', str(fps), # frame rate
                                '-s', '{}x{}'.format(canvas_width, canvas_height), # size of image string
                                '-pix_fmt', 'rgb24', # input format
                                '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
                                '-vcodec', 'h264', # output encoding
                                '-pix_fmt' ,'yuv420p', # required for some media players
                                '-vframes', str(len(range(0, nFrames, leapFactor))), # number of frames
                                outfile) # file name
                        animationProcess = subprocess.Popen(animationCmd, stdin=subprocess.PIPE)
                    
                    imgString = fig.canvas.tostring_rgb() # extract the image as an RGB string
                    if '.gif' in outfile:
                        paletteProcess.stdin.write(imgString) # write frame to GIF palette
                    animationProcess.stdin.write(imgString) # write frame to animation
                else: # use imagemagick: save frames temporarily 
                    file = filename(tmpdir, frame)
                    print('Saving frame {}/{} as "{}"'.format(frame+1, nFrames, file))
                    plt.savefig(file, facecolor=plt.gcf().get_facecolor())
                plt.close()
            if useffmpeg:
                if '.gif' in outfile:
                    paletteProcess.communicate() # Create palette
                animationProcess.communicate() # Create animation
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
