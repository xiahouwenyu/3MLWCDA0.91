# Custom color palettes for use with HEALPix maps

__version__ = "$Id: MapPalette.py 34620 2016-09-19 19:55:32Z criviere $"

try:
    import matplotlib as mpl
    import numpy as np
except ImportError as e:
    print(e)
    raise SystemExit

def setupDefaultColormap(ncolors):
    """Create a color map based on the standard blue-green-red "jet", palette.

        Args:
            ncolors: number of colors in the palette
        Returns:
            textcolor: color for text and annotation in this map
            newcm: newly defined color palette
    """
    print("ncolors =", ncolors) 

    newcm = mpl.colors.LinearSegmentedColormap("jet",
                                               mpl.cm.jet._segmentdata,
                                               ncolors)
    newcm.set_over(newcm(1.0))
    newcm.set_under("w")
    newcm.set_bad("gray")
    textcolor = "#000000"

    return textcolor, newcm


def setupCubeHelixColormap(ncolors):
    """Create a color map based on the 'cubehelix' palette.

        Args:
            ncolors: number of colors in the palette
        Returns:
            textcolor: color for text and annotation in this map
            newcm: newly defined color palette
    """
    print("ncolors =", ncolors)

    newcm = mpl.colors.LinearSegmentedColormap("cubehelix",
                                               mpl.cm.cubehelix._segmentdata,
                                               ncolors)
    newcm.set_over(newcm(1.0))
    newcm.set_under("w")
    newcm.set_bad("gray")
    textcolor = "#000000"

    return textcolor, newcm

def setupThresholdColormap(amin, amax, threshold, ncolors):
    """ Create a color map that draws all values below the threshold in
        grayscale, and everything above in the usual "jet" rainbow RGB scale.

        Args:
            amin: minimum value in color scale
            amax: maximum value in color scale
            threshold: step between grayscale and full color scale
            ncolors: number of colors in the palette
        Returns:
            textcolor: color for text and annotation in this map
            newcm: newly defined color palette
    """
    thresh = (threshold - amin) / (amax - amin)
    if threshold <= amin or threshold >= amax:
        thresh = 0.
    dthresh = 1 - thresh
    threshDict = { "blue"  : ((0.0, 1.0, 1.0),
                              (thresh, 0.6, 0.5),
                              (thresh+0.11*dthresh,  1, 1),
                              (thresh+0.34*dthresh, 1, 1),
                              (thresh+0.65*dthresh, 0, 0),
                              (1, 0, 0)),
                   "green" : ((0.0, 1.0, 1.0),
                              (thresh, 0.6, 0.0),
                              (thresh+0.09*dthresh, 0, 0),
                              (thresh+0.36*dthresh, 1, 1),
                              (thresh+0.625*dthresh, 1, 1),
                              (thresh+0.875*dthresh, 0, 0),
                              (1, 0, 0)),
                   "red"   : ((0.0, 1.0, 1.0),
                              (thresh, 0.6, 0.0),
                              (thresh+0.35*dthresh, 0, 0),
                              (thresh+0.66*dthresh, 1, 1),
                              (thresh+0.89*dthresh, 1, 1),
                              (1, 0.5, 0.5)) }
    if threshold < 0.:
        thresh = (threshold-amin)/(amax-amin)
        if threshold <= amin or threshold >= amax:
            thresh = 0.
        threshDict = { "blue"  : (
                              (0.0, .5, .5),
                              (0.11*thresh, 1, 1),
                              (0.34*thresh,  1, 1),
                              (0.65*thresh, 0, 0),
                              (thresh, 0, .6),
                              (1, 1, 0)
                              ),
                   "green" : ((0.0, 0.0, 0.0),
                              (0.09*thresh, 0, 0),
                              (0.36*thresh, 1, 1),
                              (0.625*thresh, 1, 1),
                              (0.875*thresh, 0, 0),
                              (thresh, 0, .6),
                              (1, 1, 0)),
                   "red"   : ((0.0, 0.0, 0.0),
                              (0.35*thresh, 0, 0),
                              (0.66*thresh, 1, 1),
                              (0.89*thresh, 1, 1),
                              (thresh, 0.5, .6),
                              (1, 1, 0)) }


    newcm = mpl.colors.LinearSegmentedColormap("thresholdColormap",
                                               threshDict,
                                               ncolors)
    newcm.set_over(newcm(1.0))
    newcm.set_under("w")
    newcm.set_bad("gray")
    textcolor = "#000000"

    return textcolor, newcm

def setupAbsThresholdColormap(amin, amax, threshold, ncolors):
    """ Create a color map for "two-sided" thresholds.  Below the threshold,
        the map is a cool green-blue palette.  Between the lower and upper
        threshold, the map is gray-white-gray.  Above the upper threshold,
        the map is a warm red-yellow palette.

        Args:
            amin: minimum value in color scale
            amax: maximum value in color scale
            threshold: absolute value of step between grayscale and color scales
            ncolors: number of colors in the palette
        Returns:
            textcolor: color for text and annotation in this map
            newcm: newly defined color palette
    """
    x1 = (-threshold - amin) / (amax - amin)
    x3 = (amax - threshold) / (amax - amin)
    x2 = 1. - x1 - x3
    gvl = 0.5
    threshDict = {
        "red"    : ((0.0, 1.0, 0.5), (x1, 0.0, gvl), (x1 + 0.5*x2, 1.0, 1.0),
                    (x1 + x2, gvl, 0.7), (1.0, 1.0, 1.0)),
        "green"  : ((0.0, 1.0, 1.0), (x1, 0.0, gvl), (x1 + 0.5*x2, 1.0, 1.0),
                    (x1 + x2, gvl, 0.0), (1.0, 1.0, 1.0)),
        "blue"   : ((0.0, 1.0, 1.0), (x1, 0.7, gvl), (x1 + 0.5*x2, 1.0, 1.0),
                    (x1 + x2, gvl, 0.0), (1.0, 0.5, 1.0)) }

    newcm = mpl.colors.LinearSegmentedColormap("thresholdColormap",
                                               threshDict,
                                               ncolors)
    newcm.set_over(newcm(1.0))
    newcm.set_under("w")
    newcm.set_bad("gray")
    textcolor = "#000000"

    return textcolor, newcm

def setupCividisColormap(amin, amax, threshold, ncolors):

    textcolor = "#000000"

    cividis = mpl.cm.get_cmap(name='cividis')

    threshCividis = []

    threshMap = 0.2

    threshold2 = threshold + 0.4*(amax-threshold)
    threshMap2 = 0.8
    
    for x in np.linspace(0,1,ncolors):

        if x <= threshMap:
            y = (amin + (threshold-amin)*(x - 0)/(threshMap- 0) - amin)/(amax-amin)
        elif x <= threshMap2 :
            y = (threshold + (threshold2-threshold)*(x-threshMap)/(threshMap2-threshMap) - amin)/(amax-amin)    
        else:
            y = (threshold2 + (amax-threshold2)*(x-threshMap2)/(1-threshMap2) - amin)/(amax-amin)    
            
        threshCividis.append((y , cividis(x)))
    
    newcm = mpl.colors.LinearSegmentedColormap.from_list("threshCividis",
                                               threshCividis,
                                               ncolors)
    
    return textcolor, newcm

def setupViridisColormap(amin, amax, threshold, ncolors):

    textcolor = "#000000"

    viridis = mpl.cm.get_cmap(name='viridis')

    threshViridis = []

    threshMap = 0.2

    threshold2 = threshold + 0.4*(amax-threshold)
    threshMap2 = 0.8
    
    for x in np.linspace(0,1,ncolors):

        if x <= threshMap:
            y = (amin + (threshold-amin)*(x - 0)/(threshMap- 0) - amin)/(amax-amin)
        elif x <= threshMap2 :
            y = (threshold + (threshold2-threshold)*(x-threshMap)/(threshMap2-threshMap) - amin)/(amax-amin)    
        else:
            y = (threshold2 + (amax-threshold2)*(x-threshMap2)/(1-threshMap2) - amin)/(amax-amin)    
            
        threshViridis.append((y , viridis(x)))
    
    newcm = mpl.colors.LinearSegmentedColormap.from_list("threshViridis",
                                               threshViridis,
                                               ncolors)
    
    return textcolor, newcm

def setupMagmaColormap(amin, amax, threshold, ncolors):

    textcolor = "#ffffff"

    magma = mpl.cm.get_cmap(name='magma')

    threshMagma = []

    threshMap = 0.2

    threshold2 = threshold + 0.4*(amax-threshold)
    threshMap2 = 0.8
    
    for x in np.linspace(0,1,ncolors):

        if x <= threshMap:
            y = (amin + (threshold-amin)*(x - 0)/(threshMap- 0) - amin)/(amax-amin)
        elif x <= threshMap2 :
            y = (threshold + (threshold2-threshold)*(x-threshMap)/(threshMap2-threshMap) - amin)/(amax-amin)    
        else:
            y = (threshold2 + (amax-threshold2)*(x-threshMap2)/(1-threshMap2) - amin)/(amax-amin)    
            
        threshMagma.append((y , magma(x)))
    
    newcm = mpl.colors.LinearSegmentedColormap.from_list("threshMagma",
                                               threshMagma,
                                               ncolors)
    
    return textcolor, newcm



def setupMilagroColormap(amin, amax, threshold, ncolors):
    """ Create a color map that draws all values below the threshold in
        grayscale, and everything above in the Milagro-style
        red-yellow-green-blue-black palette.

        Args:
            amin: minimum value in color scale
            amax: maximum value in color scale
            threshold: step between grayscale and color scales
            ncolors: number of colors in the palette
        Returns:
            textcolor: color for text and annotation in this map
            newcm: newly defined color palette
    """
    thresh = (threshold - amin) / (amax - amin)
    if threshold <= amin or threshold >= amax:
        thresh = 0.
    dthresh = 1 - thresh
    threshDict = { "blue"  : ((0.0, 1.0, 1.0),
                              (thresh, 0.5, 0.5),
                              (thresh+0.077*dthresh, 0, 0),
                              (thresh+0.462*dthresh, 0, 0),
                              (thresh+0.615*dthresh, 1, 1),
                              (thresh+0.692*dthresh, 1, 1),
                              (thresh+0.769*dthresh, 0.6, 0.6),
                              (thresh+0.846*dthresh, 0.5, 0.5),
                              (thresh+0.923*dthresh, 0.1, 0.1),
                              (1, 0, 0)),
                   "green" : ((0.0, 1.0, 1.0),
                              (thresh, 0.5, 0.5),
                              (thresh+0.077*dthresh, 0, 0),
                              (thresh+0.231*dthresh, 0, 0),
                              (thresh+0.308*dthresh, 1, 1),
                              (thresh+0.385*dthresh, 0.8, 0.8),
                              (thresh+0.462*dthresh, 1, 1),
                              (thresh+0.615*dthresh, 0.8, 0.8),
                              (thresh+0.692*dthresh, 0, 0),
                              (thresh+0.846*dthresh, 0, 0),
                              (thresh+0.923*dthresh, 0.1, 0.1),
                              (1, 0, 0)),
                   "red"   : ((0.0, 1.0, 1.0),
                              (thresh, 0.5, 0.5),
                              (thresh+0.077*dthresh, 0.5, 0.5),
                              (thresh+0.231*dthresh, 1, 1),
                              (thresh+0.385*dthresh, 1, 1),
                              (thresh+0.462*dthresh, 0, 0),
                              (thresh+0.692*dthresh, 0, 0),
                              (thresh+0.769*dthresh, 0.6, 0.6),
                              (thresh+0.846*dthresh, 0.5, 0.5),
                              (thresh+0.923*dthresh, 0.1, 0.1),
                              (1, 0, 0)) }

    newcm = mpl.colors.LinearSegmentedColormap("thresholdColormap",
                                               threshDict,
                                               ncolors)
    newcm.set_over(newcm(1.0))
    newcm.set_under("w")
    newcm.set_bad("gray")
    textcolor = "#000000"

    return textcolor, newcm

def setupGammaColormap(ncolors):
    """Blue/purple/red/yellow color map used by Fermi, HESS, VERITAS, etc.

        Args:
            ncolors: number of colors in the palette
        Returns:
            textcolor: color for text and annotation in this map
            newcm: newly defined color palette
    """
    cdict = {
        "red"   : [(0.00,    0,    0),
                   (0.23,    0,    0),
                   (0.47,  0.3,  0.3),
                   (0.70,    1,    1),
                   (0.94,    1,    1),
                   (1.00,    1,    1)],
        "green" : [(0.00,    0,    0),
                   (0.23,    0,    0),
                   (0.47,  0.1,  0.1),
                   (0.70,    0,    0),
                   (0.94,    1,    1),
                   (1.00,    1,    1)],
        "blue"  : [(0.00, 0.25, 0.25),
                   (0.23, 0.55, 0.55),
                   (0.47, 0.85, 0.85),
                   (0.70,    0,    0),
                   (0.94,    0,    0),
                   (1.00,0.875,0.875)]
    }

    newcm = mpl.colors.LinearSegmentedColormap("gammaColorMap", cdict, ncolors)
    newcm.set_over(newcm(1.0))
    newcm.set_under("w")
    newcm.set_bad("gray")
    textcolor = "#00FF00"
    textcolor = "#000000"

    return textcolor, newcm

def setupFrenchColormap(ncolors):
    """Blue, white, red

        Args:
            ncolors: number of colors in the palette
        Returns:
            textcolor: color for text and annotation in this map
            newcm: newly defined color palette
    """

    sat = 0.8
    cdict = {
        "red"   : [(0.00,  0.0,  0.0),
                   (0.50,  1.0,  1.0),
                   (1.00,  sat,  sat)],
        "green" : [(0.00,  0.0,  0.0),
                   (0.50,  1.0,  1.0),
                   (1.00,  0.0,  0.0)],
        "blue"  : [(0.00,  sat,  sat),
                   (0.50,  1.0,  1.0),
                   (1.00,  0.0,  0.0)]
    }

    newcm = mpl.colors.LinearSegmentedColormap("gammaColorMap", cdict, ncolors)
    newcm.set_over(newcm(1.0))
    newcm.set_under("w")
    newcm.set_bad("gray")
    textcolor = "#000000"

    return textcolor, newcm

color1 = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2", "#BEB8DC", "#E7DAD2"]
color2 = ["#2878b5","#9ac9db","#f8ac8c","#c82423","#ff8884"]
color3 = ["#F27970","#BB9727","#54B345","#32B897","#05B9E2","#8983BF","#C76DA2"]
color4 = ["#A1A9D0","#F0988C","#B883D4","#9E9E9E","#CFEAF1","#C4A5DE","#F6CAE5","#96CCCB"]
color6 = ["#63b2ee", "#76da91", "#f8cb7f", "#f89588", "#7cd6cf", "#9192ab", "#7898e1", "#efa666", "#eddd86", "#9987ce", "#63b2ee", "#76da91"]
color7 = ["#3b6291", "#943c39", "#779043", "#624c7c", "#388498", "#bf7334", "#3f6899", "#9c403d", "#7d9847", "#675083", "#3b8ba1", "#c97937"]
color8 = ["#002c53", "#ffa510", "#0c84c6", "#ffbd66", "#f74d4d", "#2455a4", "#41b7ac"]
blue_palette = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
red_palette = ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
green_palette = ['#edf8e9', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
orange_palette = ['#fff5eb', '#fee6ce', '#fdd0a2', '#fdae6b', '#fd8d3c', '#f16913', '#d94801', '#a63603', '#7f2704']
colorrainbow = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']

import matplotlib.colors as mcolors
tabcolor = mcolors.TABLEAU_COLORS
csscolor = mcolors.CSS4_COLORS
colorall = color1+color2+color3+color4+color6+color6+color8+list(tabcolor.values())+list(csscolor.values())

