# Copied by Siddharth from: "Z:\home\mwoletz\python\prfpy\quiver.py"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 13:11:57 2021

@author: mwoletz
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree
from .prfdata import PRFMeasurement
from .utils import getArrowPath, getScatterPath, normalised_direction
from reportlab.lib.units import inch, cm
import reportlab.lib.colors as rcolors
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
import matplotlib as mpl

def default_axis_properties():
    return {
        'linewidth': 1,
        'tick_length': 4,
        'nticks': 9,
        'offset': 10,
        'color': rcolors.black,
        'grid_color': rcolors.grey,
        'grid_linewidth': 0.3,
        'tick_font_size': 10,
        'tick_label_margin': 4,
        'help_line_spacing': 2.,
        'scatter_size': 4.0}

def default_colorbar_properties():
    return {
        'nstops': 64,
        'rel_height': 2.0,
        'width': 16,
        'nticks': 6,
        'offset': 10,
        'tick_length': 4,        
        'tick_font_size': 10,
        'tick_label_margin': 2,
        'label_font_size': 14,
        'label_margin': 12,
        'colormap': 'plasma',        
        'insignificant_color': rcolors.grey}

def default_arrow_properties():
    return {
        'width': 2.0, 
        'head_length': 6.5,
        'head_width': 4.5,
        'notch': 1.0}

def default_grid_properties():
    return {
        'font': 'Georgia',
        'label_font_size': 18,
        'label_leading': 14,
        'row_spacing': 0.75*cm,
        'column_spacing': 0.75*cm,
        'label_margin': 0.75*cm,
        'default_font_size': 12,
        'seperator_linewidth': 0.5,
        'seperator_color': rcolors.black
        }

class SamplingGrid(object):
    def __init__(self, x, y):
        '''
        A two dimensional sampling grid.

        Parameters
        ----------
        x : array
            The x coordinates of the sampling points.
        y : array
            The y coordinates of the sampling points.

        Returns
        -------
        None.

        '''
        self.__x = np.array(x)
        self.__y = np.array(y)
        self.__X = np.vstack((self.__x.flatten(), self.__y.flatten())).T
        
    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y
    
    @property
    def X(self):
        return self.__X
    
    @property
    def sampling_spacing(self):
        # compute the sampling spacing as the mean euclidean distance to the closest neighbour
        tree = KDTree(self.X)
        neighbour_distance, neighbour_id = tree.query(self.X, 2) # get the two closest neighbours, since the closest one will be the point itself
        
        return neighbour_distance[:,1].mean()
    
    def distances(self, Y, metric = 'euclidean'):
        if len(Y) > 0:
            return pairwise_distances(self.X, Y, metric=metric)
        else:
            return np.ones((len(self.X), len(Y))) * np.inf
    
    def weights(self, Y, h = 1.0, metric = 'euclidean', kernel = 'gaussian', return_weights_sum=False):
        D = self.distances(Y, metric)
        
        kernel = kernel.lower()
        
        if kernel == 'gaussian':
            W = np.exp(-D**2 / (2.*h**2))
        elif kernel == 'tophat':
            W = np.float32(D < h)
        elif kernel == 'epanechnikov':
            W = np.zeros(D.shape)
            mask = D < h
            W[mask] = 1.0 - D[mask] * D[mask] / (h**2)
        elif kernel == 'exponential':
            W = np.exp(-D / h)
        elif kernel == 'linear':
            mask = D < h
            W = np.zeros(D.shape)
            W[mask] = 1 - D[mask] / h
        elif kernel == 'cosine':
            mask = D < h
            W = np.zeros(D.shape)
            W[mask] = np.cos(np.pi * D[mask] / (2.*h))
        elif kernel == 'hist':
            i_max = np.argmin(D, 0)
            W = np.zeros(D.shape)
            W[i_max, np.arange(D.shape[1])] = 1.
        
        W_sum = W.sum(1)
        W = np.divide(W, W_sum[:,None], out=np.zeros_like(W), where=W_sum[:,None]!=0)
        
        if not return_weights_sum:
            return W
        else:
            return W, W_sum
    
    @classmethod
    def radial_sampling(cls, max_radius = 8, radial_spacing = 1., angular_factor = 10):
        '''
        Creates a sampling where all samples lie on circles and the number of samples per circle increases
        with the circle radius.

        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.
        max_radius : float, optional
            The maximal radius to sample. The default is 8.
        radial_spacing : float, optional
            The spacing of the sampling rings. The default is 1..
        angular_factor : float, optional
            The factor that determines the number of sampling points per sampling radius. The default is 10.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        # the radii at which to sample
        R = np.arange(0, max_radius+radial_spacing, radial_spacing)
        
        # the number of samples per radius is a function of the radius
        N = [max(1,int(r*angular_factor)) for r in R]
        
        # create the samples as complex numbers (makes the rotation easier)
        sg_complex = [R[i]*np.exp(1j*2.*np.pi/N[i]*j) for i in range(len(R)) for j in range(N[i])]
        
        # get the samplex in x and y as the real an imaginary parts of the samples
        sx = [sgc.real for sgc in sg_complex]
        sy = [sgc.imag for sgc in sg_complex]
        
        return cls(sx, sy)
    
    @classmethod
    def sunflower_spiral_sampling(cls, max_radius = 8, number_of_points = 100):
        '''
        Creates a sampling grid where all samples are on spirals similar to the placement of 
        sunflower seeds (Fibonacci).

        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.
        max_radius : float, optional
            The maximal radius to sample. The default is 8.
        number_of_points : int, optional
            The total number of sampling points. The default is 100.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        # taken from https://stackoverflow.com/a/44164075
        indices = np.arange(0, number_of_points, dtype=float) + 0.5
        
        r = np.sqrt(indices/number_of_points)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = max_radius * r * np.cos(theta)
        y = max_radius * r * np.sin(theta)
        
        return cls(x, y)
    
    @classmethod
    def hexagonal_sampling(cls, max_radius = 8, spacing = 0.5):
        '''
        Creates a sampling grid where the samples are on a regular hexagonal grid.

        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.
        max_radius : float, optional
            The maximal radius to sample. The default is 8.
        spacing : TYPE, optional
            The spacing between neighbouring points. The default is 0.5.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        # create linear sampling from -max_radius to max_radius with spacing distance
        N_samples = int(np.ceil(max_radius * 3 / spacing)) # make it a bit larger, so it will encompass everything after shearing
        N_samples = N_samples - np.mod(N_samples, 2) + 1 # make the number uneven
        d = (np.arange(N_samples) - (N_samples-1)/2) * spacing
        
        # create transformation matrix, that will create the hexagonal sampling, see: https://www.cg.tuwien.ac.at/research/vis/VolVis/BccGrids/ORVS/html/node2.htm
        T1 = 1
        T2 = np.sqrt(3)/2*T1
        V = np.array([[T1, T1/2], [0, T2]])
        
        # create the sampling points
        i,j = np.meshgrid(d, d, indexing='ij')
        I = np.vstack((i.flatten(), j.flatten()))
        
        # rotate them to hexagonal spacing
        X = V @ I
        
        x = X[0]
        y = X[1]
        
        # filter out values that are beyond the maximal radius (the sampling looks like a diamond until now)
        mask = np.sqrt(x**2 + y**2) <= max_radius
        
        x = x[mask]
        y = y[mask]
        
        return cls(x,y)

class MeasurementPair(object):
    def __init__(self, from_measurement : PRFMeasurement, to_measurement : PRFMeasurement, p_values = None,
                 max_radius = 8., plot_radius = 2.*inch, xy_limit = None, xaxis=True, yaxis=True, xaxis_tick_labels = True, yaxis_tick_labels = True, help_lines=True,
                 axis_properties = {}, arrow_properties = {}, colorbar_properties = {}, colour_metric = 'size', significance_level = 0.05, weights = None, colour_limits = None,
                 plot_scatter = False, colorbar_label = None, custom_color_values = None):
        self.__from_measurement = from_measurement
        self.__to_measurement   = to_measurement
        self.__p_values         = p_values
        
        self.max_radius = max_radius
        self.plot_radius = plot_radius
        self.xy_limit  = xy_limit or max_radius
        self.draw_xaxis = xaxis
        self.draw_yaxis = yaxis
        self.draw_xaxis_tick_labels = xaxis_tick_labels
        self.draw_yaxis_tick_labels = yaxis_tick_labels
        self.draw_help_lines = help_lines
                
        self.__axis_properties = default_axis_properties()        
        self.__axis_properties.update(axis_properties)
        
        self.__arrow_properties = default_arrow_properties()
        self.__arrow_properties.update(arrow_properties)
        
        self.__colorbar_properties = default_colorbar_properties()
        self.__colorbar_properties.update(colorbar_properties)
        
        self.__colour_limits = colour_limits
        self.__colorbar_label = colorbar_label
        self.__custom_color_values = custom_color_values
        
        self.colour_metric      = colour_metric
        self.significance_level = significance_level
        self.plot_scatter       = plot_scatter
        
        self.weights = weights
        
    @property
    def from_measurement(self):
        return self.__from_measurement
    
    @property
    def to_measurement(self):
        return self.__to_measurement
    
    @property
    def delta_x(self):
        return self.to_measurement.x - self.from_measurement.x
    
    @property
    def delta_y(self):
        return self.to_measurement.y - self.from_measurement.y
    
    @property
    def magnitude(self):
        return np.sqrt(self.delta_x**2 + self.delta_y**2)
    
    @property
    def direction(self):
        return np.arctan2(self.delta_y, self.delta_x)
    
    @property
    def p_values(self):
        if self.has_p_values:
            return self.__p_values
        else:
            return np.zeros(self.from_measurement.x.shape)
    
    @property
    def has_p_values(self):
        return self.__p_values is not None
    
    @property
    def max_radius(self):
        return self.__max_radius
    
    @max_radius.setter
    def max_radius(self, value):
        self.__max_radius = np.abs(float(value))
        
    @property
    def plot_radius(self):
        return self.__plot_radius
    
    @plot_radius.setter
    def plot_radius(self, value):
        self.__plot_radius = np.abs(float(value))
        
    @property
    def xy_limit(self):
        return self.__xy_limit
    
    @xy_limit.setter
    def xy_limit(self, value):
        self.__xy_limit = np.abs(float(value))
        
    @property
    def draw_xaxis(self):
        return self.__draw_xaxis
    
    @draw_xaxis.setter
    def draw_xaxis(self, value):
        self.__draw_xaxis = bool(value)
        
    @property
    def draw_yaxis(self):
        return self.__draw_yaxis
    
    @draw_yaxis.setter
    def draw_yaxis(self, value):
        self.__draw_yaxis = bool(value)
        
    @property
    def draw_xaxis_tick_labels(self):
        return self.__draw_xaxis_tick_labels
    
    @draw_xaxis_tick_labels.setter
    def draw_xaxis_tick_labels(self, value):
        self.__draw_xaxis_tick_labels = bool(value)
        
    @property
    def draw_yaxis_tick_labels(self):
        return self.__draw_yaxis_tick_labels
    
    @draw_yaxis_tick_labels.setter
    def draw_yaxis_tick_labels(self, value):
        self.__draw_yaxis_tick_labels = bool(value)
        
    @property
    def draw_help_lines(self):
        return self.__draw_help_lines
    
    @draw_help_lines.setter
    def draw_help_lines(self, value):
        self.__draw_help_lines = bool(value)
        
    @property
    def axis_properties(self):
        return self.__axis_properties
    
    @property
    def arrow_properties(self):
        return self.__arrow_properties
    
    @property
    def colorbar_properties(self):
        return self.__colorbar_properties
    
    @property
    def axis_scaling(self):
        return self.plot_radius / self.max_radius
    
    @property
    def max_tick(self):
        return np.floor(np.abs(self.xy_limit))
    
    @property
    def axis_ticks(self):
        return np.unique(np.round(np.linspace(-self.max_tick, self.max_tick, self.axis_properties['nticks']), 2))
    
    @property
    def axis_tick_strings(self):
        return [f'{tick:f}'.rstrip('0').rstrip('.') for tick in self.axis_ticks]
    
    @property
    def significance_level(self):
        return self.__significance_level
    
    @significance_level.setter
    def significance_level(self, value):
        self.__significance_level = np.abs(float(value))
    
    @property
    def colour_metric(self):
        return self.__colour_metric
    
    @colour_metric.setter
    def colour_metric(self, value):
        metric = str(value).lower()
        
        if metric in ['magnitude', 'size', 'variance_explained', 'angle', 'direction', 'weights', 'custom']:
            self.__colour_metric = metric
        else:
            raise ValueError(f'Metric {metric} is unsupported.')
            
    @property
    def colour_bar_limits(self):
        if self.__colour_limits is not None: # if some where specified, take them
            return self.__colour_limits
        
        if self.colour_metric == 'magnitude':
            return (0, self.max_radius*0.75)
        elif self.colour_metric == 'size':
            return (0., 2)
        elif self.colour_metric == 'variance_explained':
            return (0., 100.)
        elif self.colour_metric == 'angle':
            return (-180., 180)
        elif self.colour_metric == 'direction':
            return (-180., 180.)
        elif self.colour_metric == 'weights':
            return (0, 5)
        
    @property
    def custom_color_values(self):
        return self.__custom_color_values
    
    @custom_color_values.setter
    def color_values(self, values):
        self.__custom_color_values = values
        
    @property
    def weights(self):
        return self.__weights
    
    @weights.setter
    def weights(self, value):
        self.__weights = value
        
    @property
    def colour_bar_map(self):
        colour_limits = self.colour_bar_limits
        return mpl.cm.ScalarMappable(
                            cmap=mpl.cm.get_cmap(self.colorbar_properties['colormap']), 
                            norm=mpl.colors.Normalize(
                            vmin=colour_limits[0],
                            vmax=colour_limits[1]))
    
    @property
    def colour_bar_label_text(self):
        if self.__colorbar_label is not None:
            return self.__colorbar_label
        
        if self.colour_metric == 'magnitude':
            return 'Magnitude [°]'
        elif self.colour_metric == 'size':
            return 'pRF Size [°]'
        elif self.colour_metric == 'variance_explained':
            return 'Variance Explained [%]'
        elif self.colour_metric == 'angle':
            return 'Angle [°]'
        elif self.colour_metric == 'direction':
            return 'Angle [°]'
        elif self.colour_metric == 'weights':
            return 'Weights [a.u.]'
        
    @property
    def colour_values(self):
        if self.custom_color_values is not None:
            return self.custom_color_values
        
        if self.colour_metric == 'magnitude':
            return self.magnitude
        
        elif self.colour_metric == 'size':
            return (self.from_measurement.scale,
                    self.to_measurement.scale)
        
        elif self.colour_metric == 'variance_explained':
            return (self.from_measurement.variance_explained, self.to_measurement.variance_explained)
        
        elif self.colour_metric == 'angle':
            return (np.rad2deg(self.from_measurement.polar_angle),
                    np.rad2deg(self.to_measurement.polar_angle))
        
        elif self.colour_metric == 'direction':
            return np.rad2deg(self.direction)
        
        elif self.colour_metric == 'weights':
            return self.weights
    
    @property
    def plot_scatter(self):
        return self.__plot_scatter
    
    @plot_scatter.setter
    def plot_scatter(self, value):
        self.__plot_scatter = bool(value)
    
    def get_additional_size(self, font = 'Helvetica'): 
        width = 0.
        height = 0.
        
        if self.draw_xaxis:
            height += self.axis_properties['offset'] + self.axis_properties['tick_length']
            
            if self.draw_xaxis_tick_labels:
                height += self.axis_properties['tick_font_size'] + self.axis_properties['tick_label_margin']
        
        if self.draw_yaxis:
            width += self.axis_properties['offset'] + self.axis_properties['tick_length']
            
            if self.draw_yaxis_tick_labels:
                width += self.axis_properties['tick_label_margin']
                
                from reportlab.pdfbase.pdfmetrics import stringWidth
                
                max_tick_string_width = 0.
                
                for tick_string in self.axis_tick_strings:
                    max_tick_string_width = max(stringWidth(tick_string, font, self.axis_properties['tick_font_size']), max_tick_string_width)
                
                width += max_tick_string_width
                
        return width, height
        
    def get_total_axis_size(self, font = 'Helvetica'):
        width, height = self.get_additional_size(font)
        
        width  += 2. * self.xy_limit * self.axis_scaling
        height += 2. * self.xy_limit * self.axis_scaling
        
        return width, height
                
    
    def drawOn(self, can, x, y):
        # draws the measurement pair as a quiver plot with the radial centre at x, y
        
        can.saveState()
        # can.translate(x, y) # move the canvas to the centre of the plot # WARNING: this introduced bug in reportlab, where color gradients didn't work if a coordinate was close to 0.
        
        def ax2page(p):
            return p[0] * self.axis_scaling + x, p[1] * self.axis_scaling + y
        
        # always draw a circle about the figure
        can.saveState()
        can.setStrokeColor(self.axis_properties['color'])            
        can.circle(x, y, self.plot_radius)
        can.restoreState()
        
        # draw help line if they are to be drawn
        if self.draw_help_lines:
            can.saveState()
            can.setStrokeColor(self.axis_properties['grid_color'])
            can.setLineWidth(self.axis_properties['grid_linewidth'])
            
            # draw circles every 'help_line_spacing' (e.g.: 2) degrees, starting at 'help_line_spacing' (e.g.: 2) degrees
            for r in np.arange(self.axis_properties['help_line_spacing'], self.max_radius, self.axis_properties['help_line_spacing']):
                can.circle(x, y, r * self.axis_scaling)
            
            # draw the four lines corresponding to the horizontal, vertical and both diagonals
            for a in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                gz = self.max_radius * np.exp(1j*a)
                
                g0 = ax2page(( gz.real,  gz.imag))
                g1 = ax2page((-gz.real, -gz.imag))
                
                can.line(g0[0], g0[1], g1[0], g1[1])
                
            can.restoreState()
            
        # draw x-axis
        if self.draw_xaxis:
            can.saveState()
            can.setStrokeColor(self.axis_properties['color'])
            can.setLineWidth(self.axis_properties['linewidth'])
            
            axp1 = ax2page(( -self.max_tick, -self.xy_limit))
            axp2 = ax2page((  self.max_tick, -self.xy_limit))
            
            # draw the x-axis line            
            can.line(axp1[0]-self.axis_properties['linewidth']*0.5,
                     axp1[1]-self.axis_properties['offset'],
                     axp2[0]+self.axis_properties['linewidth']*0.5,
                     axp2[1]-self.axis_properties['offset'])
            
            # draw the ticks
            for tick, tick_string in zip(self.axis_ticks, self.axis_tick_strings):
                tp = ax2page((tick, -self.xy_limit))
                can.line(tp[0],
                         tp[1] - self.axis_properties['offset'],
                         tp[0], 
                         tp[1] - self.axis_properties['offset'] - self.axis_properties['tick_length'])
                
                if self.draw_xaxis_tick_labels:
                    can.setFontSize(self.axis_properties['tick_font_size'])
                    can.drawCentredString(
                        tp[0],
                        tp[1] - self.axis_properties['offset'] - self.axis_properties['tick_length'] - self.axis_properties['tick_font_size'] - self.axis_properties['tick_label_margin'],
                        tick_string)
            can.restoreState()
            
        # draw the y-axis
        if self.draw_yaxis:
            can.saveState()
            can.setStrokeColor(self.axis_properties['color'])
            can.setLineWidth(self.axis_properties['linewidth'])
            
            axp1 = ax2page(( -self.xy_limit, -self.max_tick))
            axp2 = ax2page(( -self.xy_limit,  self.max_tick))
            
            # draw the y-axis line 
            can.line(axp1[0]-self.axis_properties['offset'],
                     axp1[1]-self.axis_properties['linewidth']*0.5,
                     axp2[0]-self.axis_properties['offset'],
                     axp2[1]+self.axis_properties['linewidth']*0.5)
            
            # draw the ticks
            for tick, tick_string in zip(self.axis_ticks, self.axis_tick_strings):
                tp = ax2page((-self.xy_limit, tick))
                can.line(tp[0] - self.axis_properties['offset'], 
                         tp[1], 
                         tp[0] - self.axis_properties['offset'] - self.axis_properties['tick_length'], 
                         tp[1])
                
                if self.draw_yaxis_tick_labels:
                    can.setFontSize(self.axis_properties['tick_font_size'])
                    can.drawRightString(
                        tp[0] - self.axis_properties['offset'] - self.axis_properties['tick_length'] - self.axis_properties['tick_label_margin'],
                        tp[1] - 0.35 * self.axis_properties['tick_font_size'],
                        tick_string)
            can.restoreState()
        
        scalar_map = self.colour_bar_map
        
        colour_values = self.colour_values
        
        if isinstance(colour_values, tuple): # there are two colour values for this plot type
            for (x_from, y_from, x_to, y_to, p_value, colour_value_from, colour_value_to) in zip(
                    self.from_measurement.x, self.from_measurement.y,
                    self.to_measurement.x, self.to_measurement.y,
                    self.p_values, colour_values[0], colour_values[1]):
                
                from_ax = ax2page((x_from, y_from))
                to_ax   = ax2page((x_to,   y_to  ))
                
                if not self.plot_scatter:
                    p = getArrowPath(can, from_ax, to_ax, **self.arrow_properties)
                else:
                    p = getScatterPath(can, to_ax, self.axis_properties['scatter_size'])
                    direction = normalised_direction(from_ax, to_ax)
                    
                    # adapt from an to for the scatter plot for correct color gradients
                    to_ax   = to_ax + direction * self.axis_properties['scatter_size'] / 2
                    from_ax = to_ax - direction * self.axis_properties['scatter_size'] / 2
                
                if p: # if the from and to positions are the same, no path is generated
                    can.saveState()
                
                    if p_value < self.significance_level:
                        col_from = scalar_map.to_rgba(colour_value_from)
                        col_to   = scalar_map.to_rgba(colour_value_to)
                                                
                        rcol_from = rcolors.Color(*col_from)
                        rcol_to   = rcolors.Color(*col_to)
                                                
                        # make a colour gradient at the position of the arrow shaft to the arrow tip and clip it with the path
                        can.clipPath(p, stroke=0)
                        can.linearGradient(*from_ax, *to_ax,
                                           (rcol_from, rcol_to))
                    else:
                        can.setFillColor(self.colorbar_properties['insignificant_color'])                       
                        can.drawPath(p, stroke=0, fill=1)
                
                    can.restoreState()
            
        else: # arrow only has one colour in this plot type
            for (x_from, y_from, x_to, y_to, p_value, colour_value) in zip(
                    self.from_measurement.x, self.from_measurement.y,
                    self.to_measurement.x, self.to_measurement.y,
                    self.p_values, colour_values):
                
                from_ax = ax2page((x_from, y_from))
                to_ax   = ax2page((x_to,   y_to  ))
                
                if not self.plot_scatter:
                    p = getArrowPath(can, from_ax, to_ax, **self.arrow_properties)
                else:
                    p = getScatterPath(can, to_ax, self.axis_properties['scatter_size'])
                                
                if p: # if the from and to positions are the same, no path is generated
                    can.saveState()
                
                    if p_value < self.significance_level:
                        col = scalar_map.to_rgba(colour_value)
                        can.setFillColorRGB(*col)
                        can.drawPath(p, stroke=0, fill=1)
                    else:
                        can.setFillColor(self.colorbar_properties['insignificant_color'])                       
                        can.drawPath(p, stroke=0, fill=1)
                
                    can.restoreState()
        
        # undo the translation 
        # can.restoreState() # removed since using the translation introduced a bug with the reportlab linear gradient
    
    @property
    def colorbar_width(self):
        return self.colorbar_properties['width'] + 2.5*self.colorbar_properties['offset'] + \
               self.colorbar_properties['tick_label_margin'] + self.colorbar_properties['label_font_size'] + \
               self.colorbar_properties['label_margin'] + self.colorbar_properties['tick_length']
    
    def drawColorBar(self, can, x, y, height):
        
        x += self.colorbar_properties['offset']
        
        # sample the colormap nstops times. The final color bar will be a color gradient between these stops
        cbar_stops = np.linspace(*self.colour_bar_limits, self.colorbar_properties['nstops'])
        scalar_map = self.colour_bar_map
        cbar_cols = (rcolors.Color(*scalar_map.to_rgba(cbs)) for cbs in cbar_stops)
        
        # create the color gradient
        can.saveState()
        pcb = can.beginPath()
        pcb.moveTo(x, y)
        pcb.rect(x, y, self.colorbar_properties['width'], height)
        can.clipPath(pcb, stroke=0)
        can.linearGradient(x+0.5*self.colorbar_properties['width'], y, x+0.5*self.colorbar_properties['width'], y + height, cbar_cols, np.linspace(0., 1., self.colorbar_properties['nstops']).tolist())
        can.restoreState()
        
        # draw the rectangle about the gradient
        can.saveState()
        can.setStrokeColor(self.axis_properties['color'])
        can.setLineWidth(self.axis_properties['linewidth'])
        can.rect(x, y, self.colorbar_properties['width'], height)
        
        # draw the ticks
        cbar_ticks = np.linspace(*self.colour_bar_limits, self.colorbar_properties['nticks'])
        cbar_ticks_left = x + self.colorbar_properties['width']
        cbar_ticks_right = cbar_ticks_left + self.colorbar_properties['tick_length']
        max_tick_width = -1.
        
        for t, tick in enumerate(cbar_ticks):
            rt = float(t) / (self.colorbar_properties['nticks']-1.)
            tick_y = y + rt * height
            can.line(cbar_ticks_left, tick_y, cbar_ticks_right, tick_y)
            can.setFontSize(self.colorbar_properties['tick_font_size'])
            tick_string = '{:f}'.format(tick).rstrip('0').rstrip('.')
            can.drawString(cbar_ticks_right + self.colorbar_properties['tick_label_margin'], tick_y - 0.35 * self.colorbar_properties['tick_font_size'], tick_string)
            max_tick_width = max((max_tick_width, can.stringWidth(tick_string)))
        can.restoreState()
        
        # draw the label
        can.saveState()
        can.setFontSize(self.colorbar_properties['label_font_size'])
        can.translate(cbar_ticks_right + self.colorbar_properties['tick_label_margin'] + max_tick_width + self.colorbar_properties['label_margin'], y + 0.5*height)
        can.rotate(90.)
        can.drawCentredString(0., -0.35 * self.colorbar_properties['label_font_size'], self.colour_bar_label_text)
        can.restoreState()
        
    def set_mask(self, variance_explained = None, eccentricity = None, additional_mask = None):
        # generate masks for each data set
        from_measurement_mask = self.from_measurement.set_mask(variance_explained, eccentricity, additional_mask) 
        to_measurement_mask   = self.to_measurement.set_mask(  variance_explained, eccentricity, additional_mask)
        
        # combine both masks
        combined_mask = from_measurement_mask & to_measurement_mask
        
        # apply the mask to both data sets
        self.from_measurement.set_mask(additional_mask=combined_mask)
        self.to_measurement.set_mask(  additional_mask=combined_mask)
        

    def sample(self, sampling_grid : SamplingGrid, contrast = True, **kwargs):
        
        weights, weight_sum = sampling_grid.weights(self.from_measurement.X, return_weights_sum = True, **kwargs)
        
        if contrast:
            weights_to, weight_sum_to = sampling_grid.weights(self.to_measurement.X, return_weights_sum = True, **kwargs)
            
            weights = (weights + weights_to) / 2. # the contrast is numerically equal to the average of the weights
            weight_sum = (weight_sum + weight_sum_to) / 2.
            
        dx_sample = weights @ self.delta_x
        dy_sample = weights @ self.delta_y
        
        scale_from_sample = weights @ self.from_measurement.scale
        scale_to_sample   = weights @ self.to_measurement.scale
        
        variance_explained_from_sample = weights @ self.from_measurement.variance_explained
        variance_explained_to_sample   = weights @ self.to_measurement.variance_explained
        
        x_from_sample = sampling_grid.x
        y_from_sample = sampling_grid.y
        
        x_to_sample = x_from_sample + dx_sample
        y_to_sample = y_from_sample + dy_sample
        
        from_measurement_sample = PRFMeasurement(x_from_sample,     y_from_sample,
                                                 scale_from_sample, variance_explained_from_sample)
        
        to_measurement_sample   = PRFMeasurement(x_to_sample,       y_to_sample,
                                                 scale_to_sample,   variance_explained_to_sample)
        
        return MeasurementPair(from_measurement_sample, to_measurement_sample, max_radius=self.max_radius,
                               plot_radius=self.plot_radius, xy_limit=self.xy_limit, xaxis=self.draw_xaxis, yaxis=self.draw_yaxis,
                               xaxis_tick_labels=self.draw_xaxis_tick_labels, yaxis_tick_labels=self.draw_yaxis_tick_labels,
                               help_lines=self.draw_help_lines, axis_properties=self.axis_properties, arrow_properties=self.arrow_properties,
                               colorbar_properties=self.colorbar_properties, colour_metric=self.colour_metric, significance_level=self.significance_level,
                               weights = weight_sum)
    
    @classmethod
    def group_result(cls, measurement_pairs, bonferroni_correction = True, weighted=False, mu=0, **kwargs):
        '''
        Generates a measurement pair from sampled measurement pairs.

        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.
        measurement_pairs : TYPE
            A list of measurement pairs, assumed to be already sampled on the same grid.
        bonferroni_correction : bool
            If Bonferroni correction with the number of sampling points is to be performed.
        weighted : bool
            If the weights from the first level are to be used in the second level. If False, the weights
            are binarised with > 0 and used as weights. This ensures that if there was no input, this will not be counted as input.
        **kwargs : TYPE
            Any option passed to MeasurementPair.

        Returns
        -------
        None.

        '''
        from .utils import hotelling_t_square_test
        
        N_samples = measurement_pairs[0].from_measurement.N0 # assume that all pairs, including from and to have the same number of points (were sampled at the same positions)
        
        # get all values as arrays from the measurement pairs
        x_from = np.array([mp.from_measurement.x for mp in measurement_pairs])
        y_from = np.array([mp.from_measurement.y for mp in measurement_pairs])
        
        x_to = np.array([mp.to_measurement.x for mp in measurement_pairs])
        y_to = np.array([mp.to_measurement.y for mp in measurement_pairs])
        
        scale_from = np.array([mp.from_measurement.scale for mp in measurement_pairs])
        scale_to   = np.array([mp.to_measurement.scale   for mp in measurement_pairs])
        
        variance_explained_from = np.array([mp.from_measurement.variance_explained for mp in measurement_pairs])
        variance_explained_to   = np.array([mp.to_measurement.variance_explained   for mp in measurement_pairs])
        
        weights = np.array([mp.weights for mp in measurement_pairs])
        
        # compute the shift in position
        delta_x = x_to - x_from
        delta_y = y_to - y_from
        
        p_values = []
        
        individual_mu = False
        
        if not np.isscalar(mu):
            mu = np.array(mu)
            if mu.shape == (len(x_from), 2):
                individual_mu = True
        
        # test the shift for significance using a Hotelling's t²-test
        for i in range(N_samples):
            if individual_mu:
                mu_i = mu[i]
            else:
                mu_i = mu
            
            t_squared, p_value = hotelling_t_square_test(np.vstack((delta_x[:,i].flatten(), delta_y[:,i].flatten())).T, weights=(weights[:,i] if weighted else (weights[:,i] > 0).astype(float)), mu=mu_i) # always weigh at least with a binary version of the weights, so if there was no input whatsoever, this is taken into account
            
            if bonferroni_correction:
                p_values.append(p_value * N_samples) # Bonferroni correct the p-values for the number of samples
            else:
                p_values.append(p_value)
            
        p_values = np.array(p_values)
        
        # compute the average values # weighted mean option is commented out since the respective weighted test is not implemented
        if not weighted:
            x_from = np.nanmean(x_from, axis=0)
            y_from = np.nanmean(y_from, axis=0)
            
            x_to = np.nanmean(x_to, axis=0)
            y_to = np.nanmean(y_to, axis=0)
            
            scale_from = np.nanmean(scale_from, axis=0)
            scale_to   = np.nanmean(scale_to,   axis=0)
            
            variance_explained_from = np.nanmean(variance_explained_from, axis=0)
            variance_explained_to   = np.nanmean(variance_explained_to,   axis=0)
        else:        
            sum_weights = np.nansum(weights, axis=0)
            weights = np.divide(weights, sum_weights, out=np.zeros_like(weights), where=sum_weights!=0)
            
            x_from = np.nansum(x_from * weights, axis=0)
            y_from = np.nansum(y_from * weights, axis=0)
            
            x_to = np.nansum(x_to * weights, axis=0)
            y_to = np.nansum(y_to * weights, axis=0)
            
            scale_from = np.nansum(scale_from * weights, axis=0)
            scale_to   = np.nansum(scale_to * weights,   axis=0)
            
            variance_explained_from = np.nansum(variance_explained_from * weights, axis=0)
            variance_explained_to   = np.nansum(variance_explained_to * weights,   axis=0)
        
        from_measurement_sample = PRFMeasurement(x_from,     y_from,
                                                 scale_from, variance_explained_from)
        
        to_measurement_sample   = PRFMeasurement(x_to,       y_to,
                                                 scale_to,   variance_explained_to)
        
        return cls(from_measurement_sample, to_measurement_sample, p_values=p_values, weights = sum_weights if weighted else None, **kwargs)
    
    def to_offset_pair(self, target_delta_x = 0, target_delta_y = None, color_metric = 'from_eccentricity', max_radius = None, **kwargs):
        if np.isscalar(target_delta_x):
            target_delta_x = np.ones_like(self.delta_x) * target_delta_x
        
        if target_delta_y is None:
            target_delta_y = target_delta_x
            
        offset_x = self.delta_x - target_delta_x
        offset_y = self.delta_y - target_delta_y
        
        if color_metric == 'from_eccentricity':
            color_values = self.from_measurement.eccentricity
            color_label  = 'Sample eccentricity [°]'
            color_limit = (0, self.max_radius)
        elif color_metric == 'to_eccentricity':
            color_values = self.to_measurement.eccentricity
            color_label  = 'Target eccentricity [°]'
            color_limit = (0, self.max_radius)
        else:
            color_values = color_label = color_limit = None
            
        if max_radius is None:
            max_radius = np.ceil(np.max(np.sqrt(offset_x**2 + offset_y**2)))
            
        from_measurement = PRFMeasurement(np.zeros_like(self.from_measurement.x), np.zeros_like(self.from_measurement.x),
                                          self.from_measurement.scale, self.from_measurement.variance_explained)
        
        to_measurement   = PRFMeasurement(offset_x, offset_y,
                                          self.from_measurement.scale, self.from_measurement.variance_explained)
        
        plot_radius       = kwargs.get('plot_radius',       self.plot_radius)
        xaxis             = kwargs.get('xaxis',             self.draw_xaxis)
        yaxis             = kwargs.get('yaxis',             self.draw_yaxis)
        xaxis_tick_labels = kwargs.get('xaxis_tick_labels', self.draw_xaxis_tick_labels)
        yaxis_tick_labels = kwargs.get('yaxis_tick_labels', self.draw_yaxis_tick_labels)
        help_lines        = kwargs.get('help_lines',        self.draw_help_lines)
        
        significance_level = kwargs.get('significance_level', self.significance_level)
        color_limit        = kwargs.get('color_limit',             color_limit)
        
        axis_properties     = self.axis_properties.copy()
        axis_properties.update(    kwargs.get('axis_properties',     {}))
        arrow_properties    = self.arrow_properties.copy()
        arrow_properties.update(   kwargs.get('arrow_properties',    {}))
        colorbar_properties = self.colorbar_properties.copy()
        colorbar_properties.update(kwargs.get('colorbar_properties', {}))        
        
        color_metric = 'custom' if color_values is not None else self.colour_metric
        
        return MeasurementPair(from_measurement, to_measurement, p_values=self.p_values, max_radius = max_radius, plot_radius=plot_radius,
                               xaxis=xaxis, yaxis=yaxis, xaxis_tick_labels=xaxis_tick_labels, yaxis_tick_labels=yaxis_tick_labels,
                               help_lines=help_lines, axis_properties=axis_properties, arrow_properties=arrow_properties, colorbar_properties=colorbar_properties,
                               colour_metric=color_metric, significance_level=significance_level, plot_scatter=True, colour_limits=color_limit, colorbar_label=color_label, custom_color_values=color_values)
    
class QuiverGrid(object):
    def __init__(self, data, row_labels = None, column_labels = None, draw_axis = 'all', collapse_tick_labels = True, **kwargs):
        self.__N = len(data)
        
        self.__M = max([len(row) for row in data])
        
        self.__data = data
        self.__row_labels    = row_labels
        self.__column_labels = column_labels
        
        self.draw_axis = draw_axis
        self.collapse_tick_labels = collapse_tick_labels
        
        self.__grid_properties = default_grid_properties()
        self.__grid_properties.update(kwargs)
        
        self.__register_fonts()
        
    @property
    def data(self):
        return self.__data
    
    @property
    def row_label_texts(self):
        return self.__row_labels
    
    @property
    def column_label_texts(self):
        return self.__column_labels
    
    @property
    def N(self):
        return self.__N
    
    @property
    def M(self):
        return self.__M
    
    @property
    def grid_properties(self):
        return self.__grid_properties
    
    @property
    def draw_axis(self):
        return self.__draw_axis
    
    @draw_axis.setter
    def draw_axis(self, value):
        value = str(value).lower()
        if value in ['all', 'none', 'fringe']:
            self.__draw_axis = value
        else:
            raise ValueError('Unknown draw_axis option.')
            
        for r, row in enumerate(self.data):
            for c, measurement_pair in enumerate(row):
                if self.__draw_axis == 'all':
                    measurement_pair.draw_xaxis = True
                    measurement_pair.draw_yaxis = True
                elif self.__draw_axis == 'none':
                    measurement_pair.draw_xaxis = False
                    measurement_pair.draw_yaxis = False
                elif self.__draw_axis == 'fringe':
                    measurement_pair.draw_xaxis = r == (self.N-1)
                    measurement_pair.draw_yaxis = c == 0
    
    @property
    def column_ticks_consistent(self):
        tick_settings = []
        ticks_consistent = [True] * self.M
        
        for r, row in enumerate(self.data):
            for c, measurement_pair in enumerate(row):
                if len(tick_settings) > c:
                    if not (np.all(tick_settings[c][0] == measurement_pair.axis_ticks) and (tick_settings[c][1] == measurement_pair.axis_scaling)):
                        ticks_consistent[c] = False
                else:
                    tick_settings.append((measurement_pair.axis_ticks, measurement_pair.axis_scaling))
        
        return ticks_consistent
    
    @property
    def row_ticks_consistent(self):
        ticks_consistent = [True] * self.N
        
        for r, row in enumerate(self.data):
            tick_settings = None
            for c, measurement_pair in enumerate(row):
                if c == 0:
                    tick_settings = (measurement_pair.axis_ticks, measurement_pair.axis_scaling)
                else:
                    if not (np.all(tick_settings[0] == measurement_pair.axis_ticks) and (tick_settings[1] == measurement_pair.axis_scaling)):
                        ticks_consistent[r] = False
        
        return ticks_consistent   
    
    @property
    def collapse_tick_labels(self):
        return self.__collapse_tick_labels
    
    @collapse_tick_labels.setter
    def collapse_tick_labels(self, value):
        self.__collapse_tick_labels = bool(value)
        
        for r, row in enumerate(self.data):
            for c, measurement_pair in enumerate(row):
                if self.__collapse_tick_labels and self.column_ticks_consistent[c]:
                    measurement_pair.draw_xaxis_tick_labels = r == (self.N-1)
                else:
                    measurement_pair.draw_xaxis_tick_labels = True
                    
                if self.__collapse_tick_labels and self.row_ticks_consistent[r]:
                    measurement_pair.draw_yaxis_tick_labels = c == 0
                else:
                    measurement_pair.draw_yaxis_tick_labels = True
                    
    @property
    def column_offsets(self):
        column_offsets = [0] * self.M
        for r, row in enumerate(self.data):
            for c, measurement_pair in enumerate(row):
                column_offsets[c] = max(measurement_pair.xy_limit * measurement_pair.axis_scaling, column_offsets[c])
        return column_offsets
    
    @property
    def row_offsets(self):
        row_offsets = [0] * self.N
        for r, row in enumerate(self.data):
            for c, measurement_pair in enumerate(row):
                row_offsets[r] = max(measurement_pair.xy_limit * measurement_pair.axis_scaling, row_offsets[r])
        return row_offsets                
    
    @property
    def color_bar_indices(self):
        color_bar_infos = []
        color_bar_indices = []
        
        for r, row in enumerate(self.data):
            index_row = []
            for c, measurement_pair in enumerate(row):
                infos = (measurement_pair.colour_metric, 
                          measurement_pair.colour_bar_limits,
                          measurement_pair.colorbar_properties['colormap'])
                if infos in color_bar_infos:
                    index_row.append(color_bar_infos.index(infos))
                else:
                    index_row.append(len(color_bar_infos))
                    color_bar_infos.append(infos)
                    
        
            color_bar_indices.append(index_row)
            
        return color_bar_indices
    
    @property
    def color_bar_regions(self):
        indices = self.color_bar_indices
        
        row_changes = []
        
        # first find index changes in a row
        for r, row in enumerate(indices):
            last_index = -1
            row_change = []
            
            for c, index in enumerate(row):
                if index != last_index:
                    row_change.append([c, 1, index]) # we'll store the starting index, the number of consecutive columns and the index for the color bar
                    last_index = index
                else:
                    row_change[-1][1] += 1
            
            row_changes.append(row_change)
                        
        # now merge neighbouring changes across rows        
        color_bar_regions = []
        
        for r in range(len(row_changes)):
            for c in range(len(row_changes[r])):
                next_r = r
                while True:
                    next_r += 1
                    if next_r < len(row_changes):
                        if row_changes[r][c] not in row_changes[next_r]:
                            break
                        else:
                            row_changes[next_r].remove(row_changes[r][c])
                    else:
                        break
                            
                color_bar_regions.append(
                (
                    (r,        row_changes[r][c][0]),
                    (next_r-1, row_changes[r][c][0] + row_changes[r][c][1] - 1)
                ))
                
        return color_bar_regions
    
    @property
    def additional_column_spaces(self):
        color_bar_regions = self.color_bar_regions
        
        additional_spaces = [0] * self.M
        
        for c in range(self.M):
            color_bar_end_column = [cbr[1][1] for cbr in color_bar_regions]
            if c in color_bar_end_column:
                region_indices, = np.nonzero(np.array(color_bar_end_column) == c)
                
                for region_index in region_indices:
                    measurement_pair = self.data[color_bar_regions[region_index][0][0]][color_bar_regions[region_index][0][1]]
                    
                    additional_spaces[c] = max(measurement_pair.colorbar_width, additional_spaces[c])
        
        return additional_spaces
    
    def __register_fonts(self):
        # register fonts
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        if self.grid_properties['font'] == 'Georgia':
            if 'Georgia' not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont('Georgia', 'georgia.ttf'))
                pdfmetrics.registerFont(TTFont('Georgia-Bold', 'georgiab.ttf'))
                pdfmetrics.registerFont(TTFont('Georgia-Italic', 'georgiai.ttf'))
                pdfmetrics.registerFontFamily('Georgia', normal='Georgia', bold='Georgia-Bold', italic='Georgia-Italic')
                    
    @property
    def row_heights(self):
        row_heights = []
        for row in self.data:
            row_heights.append(max([measurement_pair.get_total_axis_size(self.grid_properties['font'])[1] for measurement_pair in row]))
            
        return row_heights
    
    @property
    def column_widths(self):
        column_widths = [0] * self.M
        
        for row in self.data:
            for c, measurement_pair in enumerate(row):
                column_widths[c] = max(measurement_pair.get_total_axis_size(self.grid_properties['font'])[0], column_widths[c])
            
        return column_widths
    
    @property
    def label_style(self):
        return ParagraphStyle(
                'labelsty',
                alignment = TA_CENTER,
                fontSize = self.grid_properties['label_font_size'],
                fontName = self.grid_properties['font'],
                leading  = self.grid_properties['label_leading'])
    
    @property
    def row_labels(self):
        if self.row_label_texts is not None:
            row_labels = []
            
            row_heights = self.row_heights
            
            for row_label_text, available_width in zip(self.row_label_texts, row_heights):
                label_paragraph = Paragraph(row_label_text, self.label_style)
                label_paragraph.wrap(available_width, available_width) # set the height to available width as well for square regions
                row_labels.append(label_paragraph)
                
            return row_labels
        else:
            return None
    
    @property
    def column_labels(self):
        if self.column_label_texts is not None:
            column_labels = []
            
            column_widths = self.column_widths
            
            for column_label_text, available_width in zip(self.column_label_texts, column_widths):
                label_paragraph = Paragraph(column_label_text, self.label_style)
                label_paragraph.wrap(available_width, available_width)
                column_labels.append(label_paragraph)
                
            return column_labels
        else:
            return None
        
    @property
    def column_label_height(self):
        return max([lbl.height for lbl in self.column_labels])
    
    @property
    def row_label_height(self):
        return max([lbl.height for lbl in self.row_labels])
        
    @property
    def figure_size(self):        
        width  = sum(self.column_widths) + (self.M-1) * self.grid_properties['column_spacing']
        height = sum(self.row_heights)   + (self.N-1) * self.grid_properties['row_spacing']
        
        width  += self.row_label_height    + self.grid_properties['label_margin']
        height += self.column_label_height + self.grid_properties['label_margin']
        
        width += sum(self.additional_column_spaces)
        
        return width, height
    
    def drawOn(self, can, x, y):
        fig_width, fig_height = self.figure_size
        
        column_widths = self.column_widths
        row_heights = self.row_heights
        
        additional_column_spaces = self.additional_column_spaces
        
        cummulative_widths  = np.cumsum(column_widths)
        cummulative_heights = np.cumsum(row_heights)
        
        cummulative_additional_spaces = np.cumsum(additional_column_spaces)
        
        additional_spaces = cummulative_additional_spaces - np.array(additional_column_spaces)
        
        column_offsets = self.column_offsets
        row_offsets = self.row_offsets
        
        x_offset = self.row_label_height    + self.grid_properties['label_margin']
        y_offset = self.column_label_height + self.grid_properties['label_margin']
        
        can.saveState()
        can.setFont(self.grid_properties['font'], self.grid_properties['default_font_size'])
        
        for r, row in enumerate(self.data):
            for c, measurement_pair in enumerate(row):
                x_mp = x + x_offset + cummulative_widths[c] + c * self.grid_properties['column_spacing'] - column_offsets[c] + additional_spaces[c]
                y_mp = y + fig_height - y_offset - cummulative_heights[r] + row_heights[r] - r * self.grid_properties['row_spacing'] - row_offsets[r]
                
                measurement_pair.drawOn(can, x_mp, y_mp)
        
        for r, row_label in enumerate(self.row_labels):
            can.saveState()
            can.translate(x + row_label.height, y + fig_height - y_offset - cummulative_heights[r] - r * self.grid_properties['row_spacing'] + (row_heights[r] - row_label.width) / 2.)
            can.rotate(90)
            row_label.drawOn(can, 0, 0)
            
            can.restoreState()
            
        for c, column_label in enumerate(self.column_labels):
            x_cl = x + x_offset + cummulative_widths[c] + c * self.grid_properties['column_spacing'] - column_widths[c] + (column_widths[c] - column_label.width) / 2 + additional_spaces[c]
            y_cl = y + fig_height - column_label.height
            column_label.drawOn(can, x_cl, y_cl + column_label.height/2)  # WARNING: had to add half the height for it to be placed correctly..
                
        for color_bar_region in self.color_bar_regions:
            from_ax, to_ax = color_bar_region
            
            # use the visual configurations of the first similar colorbar
            col_measurement_pair = self.data[from_ax[0]][from_ax[1]]
            
            ax_top =    y + fig_height - y_offset - cummulative_heights[from_ax[0]] + row_heights[from_ax[0]] - from_ax[0] * self.grid_properties['row_spacing']
            ax_bottom = y + fig_height - y_offset - cummulative_heights[  to_ax[0]] + row_heights[  to_ax[0]] -   to_ax[0] * self.grid_properties['row_spacing'] - row_heights[to_ax[0]]
            
            y_bot = ax_bottom + (row_heights[to_ax[0]] - row_offsets[  to_ax[0]] if (to_ax[0] != from_ax[0]) else row_heights[to_ax[0]] - 2*row_offsets[  to_ax[0]])
            y_top = ax_top    - (                        row_offsets[from_ax[0]] if (to_ax[0] != from_ax[0]) else 0.)
            
            height = y_top - y_bot
            x_col = x + x_offset + cummulative_widths[to_ax[1]] + to_ax[1] * self.grid_properties['column_spacing'] + additional_spaces[to_ax[1]]
            
            col_measurement_pair.drawColorBar(can, x_col, y_bot, height)
            
            # if the colorbar doesn't end with the last row, add a seperator line
            if to_ax[0] != self.N -1:
                y_line = ax_bottom - 0.5 * self.grid_properties['row_spacing']
                x_start = x + x_offset + cummulative_widths[from_ax[1]] - column_widths[from_ax[1]] + from_ax[1] * self.grid_properties['column_spacing'] + additional_spaces[from_ax[1]]
                x_end   = x_col + col_measurement_pair.colorbar_properties['offset'] + col_measurement_pair.colorbar_properties['width'] + col_measurement_pair.colorbar_properties['tick_length']         
                can.saveState()
                can.setLineWidth(self.grid_properties['seperator_linewidth'])
                can.setStrokeColor(self.grid_properties['seperator_color'])
                can.line(x_start, y_line, x_end, y_line)
                can.restoreState()
        
        can.restoreState()