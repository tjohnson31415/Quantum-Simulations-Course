#!/bin/bash

# Use gnuplot to create a figure showing the wavefunction squared and the
# shape of the potential well
gnuplot << TOEND
# Setting the terminal postscript with the options
set terminal postscript eps color enhanced "Helvetica" 20

set output 'plot.eps'

#Setting up labels
set title "Wavefunction Histogram"
#set xrange [0:2*pi]
set yrange [0.0:0.5]
set xlabel "Position"

# The plot itself (\ is to broke lines)
#set size 1,1
file="outputs/wavefunction.dat"
plot file index 1 u 1:2 w l lt 1 lc 3 lw 3 t "Potential", \
     file index 0 u 1:2 w l lt 1 lc 1 lw 3 t "{|{/Symbol Y}|^2}", \
     file index 2 u 1:2 w l lt 3 lc 2 lw 2 t "Theory"
TOEND

convert -density 200 plot.eps Wavefunction.png
rm plot.eps

# Create a little gif of the sample paths
gnuplot << TOEND
reset
unset key

# Set terminal to gif animate
set terminal gif animate size 1600,600 delay 50

# Setting the output file name
set output 'samplesPaths.gif'

#Setting up labels
set title "Sample Paths"
set xrange [0:2050]
set yrange [-10.0:10.0]
set xlabel "Time Step"
set ylabel "Position"

# The plot itself (\ is to broke lines)
file = "outputs/paths.dat"
set size 1,1
do for [i=0:9] {
    plot file index i u 0:1 w l lt 1 lc 3 lw 1
}

# Force the output to be written to the file
set output
TOEND
