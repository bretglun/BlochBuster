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

ABOUT
-------------------------------------------------------------------------------
BlochBuster is a nuclear magnetic resonance Bloch equation simulator written in 
Python. It simulates magnetization vectors based on the Bloch equations, 
including precession, relaxation, and excitation. BlochBuster outputs animated 
GIF files, which are 3D plots of the magnetization vectors and/or plots of 
transverse and longitudinal magnetization.  Input paramaters are provided by 
human readable configuration files. See example configuration files provided 
with BlochBuster for details.

HOW TO USE
-------------------------------------------------------------------------------
Example 1: python BlochBuster.py -c "config/SE.txt"
Example 2: python BlochBuster.py -c "config/STIR.txt" -l 5
The -c flag specifies which configuration file to use as input.
The -l flag is optional and specifies a leap factor, allowing frames to be 
skipped when generating the video file. This enables fast preview for testing.

DEPENDENCIES
-------------------------------------------------------------------------------
BlochBuster was written in Python 3.4.3, using libraries Matplotlib 1.4.3 and 
Numpy 1.9.3. BlochBuster uses ImageMagick to create animated GIF files, so
ImageMagick should be installed (www.imagemagick.org).

CONTACT INFORMATION
-------------------------------------------------------------------------------
Johan Berglund, Ph.D.
Karolinska Institutet, 
Stockholm, Sweden
johan.berglund@neuroradkarolinska.se