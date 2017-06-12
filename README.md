*Copyright (c) 2017 Johan Berglund*  
*BlochBuster is distributed under the terms of the GNU General Public License*

*This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.*

*This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.*

*You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.*

ABOUT
-----
BlochBuster is a nuclear magnetic resonance Bloch equation simulator written in Python. It simulates magnetization vectors based on the Bloch equations, including precession, relaxation, and excitation. BlochBuster outputs animated GIF files, which are 3D plots of the magnetization vectors and/or plots of transverse and longitudinal magnetization.  Input paramaters are provided by human readable configuration files. 

CONFIG FILES
------------
The configuration files are in json-format. See example configuration files provided with BlochBuster for details. 

The "pulseSeq" field is a list of RF pulses, each happening at time "t" [msec] with "FA" denoting the flip angle. Optionally, "phase" can be used to alter the phase of the RF pulse. "FA": 0 is interpreted as spoiling of transverse magnetization. The pulse sequence repeats after "TR" msec, with "nTR" repetitions. The main field "B0" is given in T, and the RF pulse duration is determined by "B1" in &mu;T. "B1"&ge;100; gives "instant" RF pulses; the clock will freeze and no relaxation or precession will occur.

The "components" field is a list of components/tissues, each represented by a magnetization vector with a distinct color in the plot. "Meq" is the vector length at equilibrium, "CS" is chemical shift in ppm, "T1" and "T2" are relaxation times in msec. Each component may be represented by a "fan" of "nIsochromats" vectors, with a distribution of precession frequencies determined by "isochromatStep" [ppm].

The animation speed is determined by the "speed" field, where 1 corresponds to real-time. The animated GIFs are saved in the /out folder. Specifying a filename for "outFile3D" will toggle a 3D animated plot of the magnetization vectors. "outFileMxy" and "outFileMz"  will toggle animated plots of transverse and longitudinal magnetization, respectively. 

HOW TO USE
----------
`Example 1: python BlochBuster.py -c "config/SE.json"`

`Example 2: python BlochBuster.py -c "config/STIR.json" -l 5 -b`

The -c flag specifies which configuration file to use as input.  
The -l flag is optional and specifies a leap factor, allowing frames to be skipped when generating the video file. This enables fast preview for testing.  
The -b flag toggles black background.

DEPENDENCIES
------------
BlochBuster was written in Python 3.4.3, using libraries Matplotlib 1.4.3 and 
Numpy 1.9.3. BlochBuster uses ImageMagick to create animated GIF files, so
ImageMagick should be installed (www.imagemagick.org).

CONTACT INFORMATION
-------------------
Johan Berglund, Ph.D.  
Karolinska Institutet,  
Stockholm, Sweden  
johan.berglund@neuroradkarolinska.se