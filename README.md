*Copyright (c) 2017-2018 Johan Berglund*  
*BlochBuster is distributed under the terms of the GNU General Public License*

*This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.*

*This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.*

*You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.*

ABOUT
-----
BlochBuster is a nuclear magnetic resonance Bloch equation simulator written in Python. 
It simulates magnetization vectors based on the Bloch equations, including precession, relaxation, and excitation. 
BlochBuster outputs animated gif or mp4 files, which can be 3D plots of the magnetization vectors, plots of transverse and longitudinal magnetization, or pulse sequence diagrams.
Input paramaters are provided by human readable configuration files.
The animations are made with ffmpeg.

CONFIG FILES
------------
The configuration files are in yaml-format. 
See example configuration file `config/example.yml` provided with BlochBuster for details. 

The "pulseSeq" field is a list of events such as RF pulses, gradients or spoiling. 
Each event happens at given time "t" [msec] with duration "dur" [msec].
An RF event may have a "FA" denoting the prescribed flip angle, or a "B1" field strength [&mu;T] that determines the FA together with the duration. 
Optionally, "phase" can be used to alter the phase of the RF pulse. 
Setting dur: 0 gives an "instant" RF pulse; the clock will freeze during flipping and no relaxation or precession will occur.
A gradient event is specified by "Gx", "Gy", and/or "Gz" [mT/m]. 
A graident may be played together with an RF-pulse, if "Gx", "Gy", and/or "Gz" is specified in the RF event.
A spoiler event is indicated by "spoil: true", and spoils all transverse magnetization. 

The pulse sequence repeats after "TR" msec, with "nTR" repetitions. The main field "B0" is given in T.

The "components" field is a list of components/tissues, each represented by a magnetization vector with a distinct color in the plot. 
Each components "name" will be given in a legend. 
"CS" is chemical shift in ppm; "T1" and "T2" are relaxation times in msec. 
Each component may be represented by a fan of "nIsochromats" vectors, with a distribution of precession frequencies determined by "isochromatStep" [ppm].

The optional "locations" field should contain a 3D matrix, indicating the equilibrium magnetization at different spatial positions.
One matrix can be given for each components, but their shapes must match.

The animation speed is determined by the "speed" field, where 1 corresponds to real-time. 

The output is specified by a list, where the "type" can be:
- 3D: animated 3D plot of the magnetization vectors
- xy: animated plot of transverse magnetization over time. If "abs: false", both real and imaginary components are plotted.
- z: animated plot of transverse magnetization over time
- psd: events plotted as a pulse sequence diagram
The filename is specified by "file". The file ending can be .gif or .mp4.

HOW TO USE
----------
`Example 1: python BlochBuster.py -c "config/SE.yml"`

`Example 2: python BlochBuster.py -c "config/SpEnc.yml" -l 5 -b`

The -c flag specifies which configuration file to use as input.  
The -l flag is optional and specifies a leap factor, allowing frames to be skipped when generating the animation file. This enables fast preview for testing.  
The -b flag toggles black background.

DEPENDENCIES
------------
BlochBuster was written in Python 3.4.3, using libraries Matplotlib 1.4.3 and 
Numpy 1.9.3. BlochBuster uses ffmpeg to create animated gif or mp4 files, so
ffmpeg needs to be installed (https://www.ffmpeg.org/).

CONTACT INFORMATION
-------------------
Johan Berglund, Ph.D.  
Karolinska Institutet,  
Stockholm, Sweden  
johan.berglund@neuroradkarolinska.se