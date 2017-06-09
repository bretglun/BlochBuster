# Copyright (c) 2015 Johan Berglund
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
import subprocess
import winreg

# Define arrow colors
color = [(0, 176/256, 80/256), 'cadetblue',
         'darkolivegreen', 'darkslateblue',
         'green', 'lightslategray', 'blue', 'brown']


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
    ax = fig.gca(projection='3d', xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    ax.set_axis_off()
    ax.set_position([-0.26, -0.39, 1.6, 1.58])

    # Draw axes circles
    for i in ["x", "y", "z"]:
        circle = Circle((0, 0), 1, fill=True, lw=1, color=(0, 0, 0, .02))
        ax.add_patch(circle)
        art3d.pathpatch_2d_to_3d(circle, z=0, zdir=i)

    # Draw x, y, and z axes
    axColor = 'gray'
    ax.plot([-1, 1], [0, 0], [0, 0], color=axColor, zorder=-1)  # x-axis
    ax.text(1.1, 0, 0, r'$x^\prime$', horizontalalignment='center')
    ax.plot([0, 0], [-1, 1], [0, 0], color=axColor, zorder=-1)  # y-axis
    ax.text(0, 1.2, 0, r'$y^\prime$', horizontalalignment='center')
    ax.plot([0, 0], [0, 0], [-1, 1], color=axColor, zorder=-1)  # z-axis
    ax.text(0, 0, 1.1, r'$z$', horizontalalignment='center')

    # Draw title:
    ax.text(0, 0, 1.4, title, fontsize=14, horizontalalignment='center')
    # Draw time
    time_text = ax.text(-1, -.8, -1, 'time = %.1f msec' % (clock[frame]))

    # Draw magnetization vectors
    nVecs = len(comps[0])
    for c in range(len(comps)):
        for m in range(nVecs):
            col = color[(c) % len(color)]
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
            color='#500000', horizontalalignment='right')
    ax.text(.7, .7, .95, RFText[frame], fontsize=14, alpha=RFTextAlpha[frame],
            color='#005000', horizontalalignment='right')
    # Draw legend:
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
                    [plt.Line2D((0, 1), (0, 0), lw=2, color=color[(c) %
                                len(color)]) for c, handle in enumerate(
                                handles)], labels, loc=2, bbox_to_anchor=[
                                .14, .83])
    leg.draw_frame(False)


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
    fig = plt.figure(figsize=(5, 2.7))
    ax = fig.gca(xlim=(xmin, xmax), ylim=(ymin, ymax))
    for side in ['bottom', 'right', 'top', 'left']:
        ax.spines[side].set_visible(False)  # remove default axes
    ax.grid()
    plt.title(title)
    plt.xlabel('time[ms]', horizontalalignment='right')
    if plotType == 'xy':
        ax.xaxis.set_label_coords(1.1, .1)
        plt.ylabel('$|M_{xy}|$', rotation=0)
    elif plotType == 'z':
        ax.xaxis.set_label_coords(1.1, .475)
        plt.ylabel('$M_z$', rotation=0)
    ax.yaxis.set_label_coords(-.07, .95)
    plt.tick_params(axis='y', labelleft='off')
    ax.xaxis.set_ticks_position('none')  # tick markers
    ax.yaxis.set_ticks_position('none')

    # draw x and y axes as arrows
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height  # get width and height of axes object
    hw = 1/25*(ymax-ymin)  # manual arrowhead width and length
    hl = 1/25*(xmax-xmin)
    yhw = hw/(ymax-ymin)*(xmax-xmin) * height/width  # compute matching arrowhead length and width
    yhl = hl/(xmax-xmin)*(ymax-ymin) * width/height
    ax.arrow(xmin, 0, (xmax-xmin)*1.05, 0, fc='k', ec='k', lw=1, head_width=hw, head_length=hl, clip_on=False)
    ax.arrow(0, ymin, 0, (ymax-ymin)*1.05, fc='k', ec='k', lw=1, head_width=yhw, head_length=yhl, clip_on=False)
    # Draw magnetization vectors
    nVecs = len(comps[0])
    if plotType == 'xy':
        Msum = np.zeros([2, frame+1])
        for c in range(len(comps)):
            col = color[(c) % len(color)]
            Mxy = np.zeros([2, frame+1])
            for m in range(nVecs):
                Mxy += comps[c][m][:2, :frame+1]
            ax.plot(clock[:frame+1], np.linalg.norm(Mxy, axis=0)/nVecs, '-', lw=2, color=col, label=names[c])
            Msum += Mxy
        # Special case: also plot sum of fat and water
        if 'water' in names and 'fat' in names:
            col = color[(len(comps)) % len(color)]
            ax.plot(clock[:frame+1], np.linalg.norm(Msum, axis=0)/nVecs/len(comps), '-', lw=2, color=col, label=names[c])
    elif plotType == 'z':
        for c in range(len(comps)):
            col = color[(c) % len(color)]
            Mz = np.zeros([frame+1])
            for m in range(nVecs):
                Mz += comps[c][m][2, :frame+1]
            ax.plot(clock[:frame+1], Mz/nVecs, '-', lw=2, color=col, label=names[c])


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


# Simulate magnetization vector during Nreps applications of pulseSeq
def applyPulseSeq(Meq, w, T1, T2, pulseSeq, w1, Nreps=1, dt=0.1, instantRF=False):
    # Initial state is equilibrium magnetization
    M = np.array([[0.], [0.], [Meq]])
    for rep in range(Nreps):
        for (FA, T, spoiler) in pulseSeq:
            # RF-pulse:
            dur = radians(abs(FA))/w1  # RF pulse duration
            t = np.arange(0, dur+dt, dt)
            w1_adj = radians(FA)/((len(t)-1)*dt)  # adjust w1 to fit FA to integer number of frames
            if instantRF:
                M1 = integrate.odeint(derivs, M[:, -1], t, args=(Meq, 0., w1_adj, np.inf, np.inf))
            else:
                M1 = integrate.odeint(derivs, M[:, -1], t, args=(Meq, w, w1_adj, T1, T2))
            M = np.concatenate((M, M1[1:].transpose()), axis=1)
            # Then relaxation
            if instantRF:
                t = np.arange(0, T+dt, dt)
            else:
                t = np.arange(0, T-dur+dt, dt)
            M1 = integrate.odeint(derivs, M[:, -1], t, args=(Meq, w, 0., T1, T2))
            M = np.concatenate((M, M1[1:].transpose()), axis=1)
            # Then spoiling
            if spoiler:
                M[:, -1] = spoil(M[:, -1])
    return M


# Simulate Nisochromats dephasing magnetization vectors of component defined by compProps
def simulateComponent(compProps, w0, Nisochromats, isochromatStep, pulseSeq, w1, Nreps=1, dt=0.1, instantRF=False):
    # Shifts in ppm for dephasing vectors:
    isochromats = [(2*i+1-Nisochromats)/2*isochromatStep+compProps[1] for i in range(0, Nisochromats)]
    comp = []
    for isochromat in isochromats:
        w = w0*isochromat*.000001  # Demodulated frequency [krad]
        comp.append(applyPulseSeq(compProps[0], w, compProps[2], compProps[3], pulseSeq, w1, Nreps, dt, instantRF))
    return comp


# Get clock during Nreps applications of pulseSeq (clock stands still during excitation)
# Get opacity and text for spoiler and RF text flashes in 3D plot
def getClockSpoilAndRFText(pulseSeq, Nreps, w1, dt, instantRF=False):
    clock = [0.0]
    decrPerFrame = .1
    spoilTextAlpha = [0.]
    RFTextAlpha = [0.]
    RFText = ['']
    for rep in range(Nreps):
        for (FA, T, spoiler) in pulseSeq:
            RF = str(int(abs(FA)))+u'\N{DEGREE SIGN}'+'-pulse'
            # TODO: add info about the RF phase angle
            # Frames during RF pulse
            dur = radians(abs(FA))/w1  # RF pulse duration
            t = np.arange(dt, dur+dt, dt)
            if instantRF:
                clock.extend(np.full(t.shape, clock[-1]))  # Clock stands still during instant RF pulse
            else:
                clock.extend(t+clock[-1])
            spoilTextAlpha.extend(np.linspace(spoilTextAlpha[-1], spoilTextAlpha[-1]-len(t)*decrPerFrame, num=len(t)))
            RFTextAlpha.extend(np.ones(len(t)))
            RFText += [RF]*len(t)
            # Frames during relaxation
            if instantRF:
                t = np.arange(dt, T+dt, dt)
            else:
                t = np.arange(dt, T-dur+dt, dt)
            clock.extend(t+clock[-1])  # Increment clock during relaxation time T
            spoilTextAlpha.extend(np.linspace(spoilTextAlpha[-1], spoilTextAlpha[-1]-len(t)*decrPerFrame, num=len(t)))
            RFTextAlpha.extend(np.linspace(RFTextAlpha[-1], RFTextAlpha[-1]-len(t)*decrPerFrame, num=len(t)))
            RFText += [RF]*len(t)
            # Spoiling
            if spoiler:
                spoilTextAlpha[-1] = 1.
    # Clip at zero
    spoilTextAlpha = [max(alpha, 0) for alpha in spoilTextAlpha]
    RFTextAlpha = [max(alpha, 0) for alpha in RFTextAlpha]
    return clock, spoilTextAlpha, RFTextAlpha, RFText


def isTrue(string): return string.lower() == 'true' or string.lower() == 'yes' or string == '1'


# Read pulse sequence from string. The pulse sequence is defined by a comma separated
# list of triples (FA,T,spoil). Each triple represents:
# 1. An RF pulse of flip angle FA degrees
# 2. Relaxation during T msec
# 3. Optional spoiling of any transverse magnetization [true/false]
def readPulseSeq(seqString):
    pulseSeq = []
    pulses = seqString.replace(' ', '')[1:-1].split('),(')
    for pulse in pulses:
        FA, T, spoil = pulse.split(',')
        pulseSeq.append((complex(FA), float(T), isTrue(spoil)))
    return pulseSeq


# Read component properties from string. The signal components (tissues) are defined by a
# comma separated list of component properties (name,Meq,shift,T1,T2):
# - name is the denomination that will be used in the figure legend (can be left blank)
# - Meq is the magnitude of the equilibrium magnetization
# - shift is the chemical shift [ppm]
# - T1 is the longitudinal relaxation time [msec]
# - T2 is the transversal relaxation time [msec]
def readCompProps(paramString):
    names = []
    compProps = []
    comps = paramString.replace(' ', '')[1:-1].split('),(')
    for comp in comps:
        name, Meq, shift, T1, T2 = comp.split(',')
        names.append(name)
        compProps.append((float(Meq), float(shift), float(T1), float(T2)))
    return names, compProps


# Read input parameters to simulation and animation from configuration file
def configParser(configFile):
    # Set default values
    title = 'Default'				# Title of animation
    pulseSeq = ((90, 10, False),)		# Gradient echo pulse sequence
    Nreps = 1 						# Number of repetitions
    names = ('',)					# List of component names
    compProps = ((1., 0., 260., 84.),)  # List of component properties (Meq,shift,T1,T2)
    B0 = 1.5             			# B0 field strength [T]
    B1 = 5             				# B1 field strength [uT]
    Nisochromats = 15 				# Number of isochromat per component
    isochromatStep = .02			# Resonance shift steplength [ppm] between isochromats
    speed = .01 					# Animation speed factor
    outfile3D = ''					# Output filename for 3D animation (should be .gif)
    outfileMxy = ''					# Output filename for transversal magnetization (should be .gif)
    outfileMz = ''					# Output filename for longitudinal magnetization (should be .gif)
    if os.path.isfile(configFile):
        for row in csv.reader(open(configFile, 'rt'), delimiter=':'):
            if row and not row[0].startswith('#'):
                if row[0] == 'Title':
                    title = row[1].strip()
                elif row[0] == 'PulseSeq':
                    pulseSeq = readPulseSeq(row[1])
                elif row[0] == 'Nreps':
                    Nreps = int(row[1])
                elif row[0] == 'CompProps':
                    names, compProps = readCompProps(row[1])
                elif row[0] == 'B0':
                    B0 = float(row[1])
                elif row[0] == 'B1':
                    B1 = float(row[1])
                elif row[0] == 'Nisochromats':
                    Nisochromats = int(row[1])
                elif row[0] == 'IsochromatStep':
                    isochromatStep = float(row[1])
                elif row[0] == 'Speed':
                    speed = float(row[1])
                elif row[0] == 'Outfile3D':
                    outfile3D = row[1].strip()
                elif row[0] == 'OutfileMxy':
                    outfileMxy = row[1].strip()
                elif row[0] == 'OutfileMz':
                    outfileMz = row[1].strip()
    return title, pulseSeq, Nreps, names, compProps, B0, B1, Nisochromats, isochromatStep, speed, outfile3D, outfileMxy, outfileMz


def filename(dir, frame): return dir + '/' + format(frame+1, '04') + '.png'


def findImageMagick():
    for baseDir in [r'C:/Program Files/', r'C:/Program Files (x86)/']:
        for dir in os.listdir(baseDir):
            subdir = baseDir + dir
            if dir.startswith('ImageMagick') and os.path.isdir(subdir):
                return subdir
    return r'.'


# Main program
def BlochBuster(configFile, leapFactor=1):
    # Prepare for using convert.exe from ImageMagick
    try:  # Windows only:
        ImageMagickKey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\ImageMagick\\Current', 0, (winreg.KEY_WOW64_64KEY + winreg.KEY_ALL_ACCESS))
        ConvertExePath = winreg.QueryValueEx(ImageMagickKey, "BinPath")[0]
    except WindowsError:
        ConvertExePath = findImageMagick()
    convert = ConvertExePath + r'\convert.exe'
    if not os.path.isfile(convert):
        raise Exception('ImageMagick:s convert.exe is required but was not found')
    compress = r'-layers Optimize'
    # Read configuration file
    title, pulseSeq, Nreps, names, compProps, B0, B1, Nisochromats, isochromatStep, speed, outfile3D, outfileMxy, outfileMz = configParser(configFile)
    instantRF = B1 >= 100		# B1=100 means instant RF pulses
    B1 = B1/1000000.			# convert uT->T
    # Calculations
    fps = 15.				# Frames per second in animation (<=15 should be supported by powepoint)
    dt = 1000./fps*speed 	# Time resolution [msec]
    gyro = 42577.			# Gyromagnetic ratio [kHz/T]
    if instantRF:
        B1 = 1/(gyro*dt*72)  # Set duration of a 360-pulse to 72 frames
    w0 = 2*np.pi*gyro*B0  # Larmor frequency [kRad]
    w1 = 2*np.pi*gyro*B1  # B1 rotation frequency [kRad]
    # Simulate
    comps = []
    for compProp in compProps:
        comps.append(simulateComponent(compProp, w0, Nisochromats, isochromatStep, pulseSeq, w1, Nreps, dt, instantRF))
    # Animate
    clock, spoilTextAlpha, RFTextAlpha, RFText = getClockSpoilAndRFText(pulseSeq, Nreps, w1, dt, instantRF)
    delay = int(100/fps*leapFactor)  # Delay between frames in ticks of 1/100 sec
    nFrames = len(comps[0][0][0])
    if not outfile3D+outfileMxy+outfileMz:
        raise Exception('No outfile (Outfile3D/OutfileMxy/OutfileMz) was found in config')
    tmpdir = r'./tmp'
    outdir = r'./out'
    if os.path.isdir(tmpdir):
        rmTmpDir = input(r'Temporary folder "{}" already exists. Delete(Y/N)?'.format(tmpdir))
        if rmTmpDir.upper() == 'Y':
            shutil.rmtree(tmpdir)
        else:
            raise Exception('No files written.')
    for (plotType, outfile) in [('3D', outfile3D), ('xy', outfileMxy), ('z', outfileMz)]:
        if outfile:
            os.mkdir(tmpdir)
            for frame in range(0, nFrames, leapFactor):
                # Use only every leapFactor frame in animation
                if plotType == '3D':
                    plotFrame3D(names, comps, title, clock, frame, spoilTextAlpha, RFTextAlpha, RFText)
                else:
                    plotFrameMT(names, comps, title, clock, frame, plotType)
                file = filename(tmpdir, frame)
                print(r'Saving frame {}/{} as "{}"'.format(frame+1, nFrames, file))
                plt.savefig(file)
                plt.close()
            if not os.path.isdir(outdir):
                os.mkdir(outdir)
            outfile = r'./out/'+outfile
            print(r'Creating animated gif "{}"'.format(outfile))
            p = subprocess.Popen('{} {} -delay {} {}/*png {}'.format(convert, compress, delay, tmpdir, outfile))
            p.wait()
            p.communicate()
            shutil.rmtree(tmpdir)


# Command line parser
def main():
    # Initiate command line parser
    p = optparse.OptionParser()
    p.add_option('--configFile', '-c', default='',  type="string", help="Name of configuration text file")
    p.add_option('--leapFactor', '-l', default=1, type="int", help="Leap factor for smaller filesize and fewer frames per second")
    # Parse command line
    options, arguments = p.parse_args()
    BlochBuster(options.configFile, options.leapFactor)

if __name__ == '__main__':
    main()
