# The current version for finding the extrema of the drop
# still needs the reflection on the surface

import numpy as np
import matplotlib.pyplot as plt
import math
import os
import glob
import sys
import imageio

#from skimage.filters import canny
from PIL import Image
from skimage.morphology import reconstruction
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage import color
from skimage import measure
from skimage.feature import canny
from scipy import misc
from scipy import ndimage
from skimage.filters import threshold_otsu
from pylab import *

debug = False # Set True for more print() info
reset90 = True # Used to check if a contact angle of over 90 deg has been found
less = 0
fileEnd = "bmp"

# Set the name for the output file according to the cwd
cwd = (os.getcwd()).split('/', 100)
fname = cwd[len(cwd)-1]

# Open file for writing
plot_file = open(fname+".dat", "w")
plot_file.write("1. filename\t2. file_number\t3. dist_left\t4. dist_right\t5. v_left\t6. v_right\t7. surface_angle\t8. droplet_width\t9. theta_left\t10. theta_right\t11. center_xs\t12. center_ys\t13. droplet_height\n")

filenumber = 0
filedif = 0
first_left = 0
first_right = 0
pre_left = 0
pre_right = 0
v_left = 0
v_right = 0
deadzone_trigger = "n"
shutter_time = 0.008333333 #in seconds/frame

YAchse = []
XAchse =[]
YAchse_gesamt = []
XAchse_gesamt =[]
Zeit = []
Tropfenhoehe = []

bb = shutter_time

Summe_A=0
Summe_A_x=0
Summe_A_y=0
x_i = []
y_i = []
A_i = []
x_s = []
y_s = []


filelist = sorted(glob.glob("*."+fileEnd))

while reset90 == True:
    if filedif != 0:
        filelist = np.delete(filelist, np.arange(filedif))
    filedif = 0
    filenumber
    for file in filelist:
        if debug:
            print("Current file: "+str(file)+"\n")
        if filenumber == 0:
            reset90 = False
        if reset90 == False:
            filenumber += 1
        # Load the image and transform to 2d-grayscale-array
        imOrig = imageio.v2.imread(file)
        ly, lx = imOrig.shape
        if deadzone_trigger == "y":
            if dz_l == 0: dz_l = 1./lx
            if dz_r == 0: dz_r = 1./lx
            if dz_t == 0: dz_t = 1./ly
            if dz_b == 0: dz_b = 1./ly
            imOrig = imOrig[ly*dz_t : -ly*dz_b, lx*dz_l : -lx*dz_r] #Crops the image
        ly, lx = imOrig.shape
        if len(imOrig.shape)==2:
            imGray = imOrig
        else:
            imGray=color.rgb2gray(imOrig)
            
        # Setting the threshold
		thresh = threshold_otsu(imGray)
		if reset90 == False: im = imGray > 0.75*thresh # > 200 can also be used, gives a small over-estimation though
		else: im = imGray > 30
		
		# Dilute bright spots
		seed = np.copy(im)
		seed[1:-1, 1:-1] = im.min()
		mask = im
		im = reconstruction(seed, mask, method='dilation')
		
		# Use a gaussian filter to smooth out edges in the background
		#im = ndimage.gaussian_filter(im, 1)
	
		# Use a median filter to smooth out edges in the background
		im = ndimage.median_filter(im, 3)
		
		# Use canny cut to get harder edges
		#im = canny(im)
				
		# Find contours at a constant value of 0.99
		drop_contour = 0 # Used to remember which contour belongs to the droplet
		max_cont_len = 0
		contours = measure.find_contours(im, 0.99, fully_connected='high', positive_orientation='low')
		if debug: print("Number of contours: "+str(len(contours))+"\n")
		for c, cont in enumerate(contours):
		  if debug: print("Length of contour #"+str(c)+" = "+str(len(cont))+"\n")
		  if max_cont_len < len(cont):
		    max_cont_len = len(cont)
		    drop_contour = c
		contours_len = len(contours[drop_contour])
		if len(contours)==0: continue
		if debug: print("Length of droplet_contour: "+str(contours_len)+"\n")	
		if debug: print("Number of droplet contour: "+str(drop_contour)+"\n")
		
		if len(contours) > 1: contours = np.delete(contours, np.s_[(drop_contour+1)::1])
		if drop_contour != 0: contours = np.delete(contours, np.s_[0:drop_contour:1])
		drop_contour = 0		
		
		if debug: print("Length of droplet contour after deleting:"+str(contours_len)+"\n")
		
		surf_min = -111
		surf_max = -111
		rot_contours = list()
		for p, point in enumerate(contours[drop_contour]): # Necessary because contours is list(array(array(list))) ... 
		  rot_contours.append([point[0], point[1]])
		  if abs(((contours[drop_contour])[0])[0]-((contours[drop_contour])[contours_len-1])[0]) > 30:
		    if point[0] > surf_min: # Find the lowest point in contour for the correct determination of the surface angle
		      surf_min = point[0]
		      surf_min_point = p
		    if point[0] < surf_max:
		      surf_max = point[0]
		      surf_max_point = p
		if surf_min != -111: rot_contours = np.delete(rot_contours, np.arange(surf_min_point-50, contours_len), 0)
		contours_len = len(rot_contours)
		# Determine the angle of the surface = alpha_surf (in DEGREES)
		surface_x = [((contours[drop_contour])[0])[1], ((contours[drop_contour])[contours_len-1])[1]]
		surface_y = [((contours[drop_contour])[0])[0], ((contours[drop_contour])[contours_len-1])[0]]
		if debug: print("surface_x: "+str(surface_x)+"\n")
		if debug: print("surface_y: "+str(surface_y)+"\n")
		horizontal = abs(((contours[drop_contour])[0])[1]-((contours[drop_contour])[contours_len-1])[1])
		vertical = abs(((contours[drop_contour])[0])[0]-((contours[drop_contour])[contours_len-1])[0])
		alpha_surf_rad = abs(math.atan(vertical/horizontal))
		alpha_surf = math.degrees(alpha_surf_rad)
		if debug: print("alpha_surf = "+str(alpha_surf)+"\n")
		if alpha_surf < 0.4: alpha_surf_rad = math.radians(0.4) # A minimal tilt to improve the tangent fit
	
		
		
		# Rotate the contour by alpha_surf
		last = rot_contours[contours_len-1]
		if debug: print("LAST "+str(last[1])+" "+str(last[0])+"\n")
		if debug: print("LX LY "+str(lx)+" "+str(ly)+"\n")
		for p, point in enumerate(contours[drop_contour]):
			if p < contours_len:
				(rot_contours[p])[1] = math.cos(alpha_surf_rad)*(point[1]-last[1])-math.sin(alpha_surf_rad)*(point[0]-last[0])+last[1]
				(rot_contours[p])[0] = math.sin(alpha_surf_rad)*(point[1]-last[1])+math.cos(alpha_surf_rad)*(point[0]-last[0])+last[0]
		rot_surface_y = math.sin(alpha_surf_rad)*(surface_x-last[1])+math.cos(alpha_surf_rad)*(surface_y-last[0])+last[0]
		
		
		# -------------------------------------
		# Process for finding the extreme values for the width of the drop
		# and the two contact angles (theta_right and theta_left)	
		# -------------------------------------
		
		# Some prerequisites
		max_right_x = 0 # The "right" is a reminder for the algorithm going from the right
		max_right_y = 0
		min_right_x = 99999999
		min_right_y = 99999999
		right_point = 0 # Memorizes the position of the extreme_values
		left_point = 0
		theta_right = -1
		theta_left = -1
		fit_points_right = 50 # Determines the amount of next points for the tangent_fit	
		fit_points_left = 50
		
		# Determine the height of the zero-line for cutting of all values that don't belong to the droplet
		zero_line = 0
		cut_off_height = 20		
		zero_line = (rot_surface_y[1]+rot_surface_y[0])/2
		if debug: print("zero_line = "+str(zero_line)+"\n")
                
		# Find the extreme left and right values of the droplet
		for p, point in enumerate(rot_contours):
			if point[0] < zero_line-cut_off_height: # "-" and "<" because the origin is in the top left corner
				if point[1] > max_right_x:
					max_right_x = point[1]
					max_right_y = point[0]
					right_point = p
				if point[1] < min_right_x:
					min_right_x = point[1]
					min_right_y = point[0]
					left_point = p
		if debug:
                	print("max_x = "+str(max_right_x)+"   max_y = "+str(max_right_y)+"\n")
                	print("min_x = "+str(min_right_x)+"   min_y = "+str(min_right_y)+"\n")					
                
                    
		# Correction for angles above 90 degrees
		over90_left = False
		over90_right = False
		left_corrector = list()
		right_corrector = list()
		for p, point in enumerate(rot_contours):
		  if point[1] > min_right_x and point[1] < (min_right_x+50) and point[0] > min_right_y and point[0] < zero_line-5:
		    left_corrector.append([point[0], point[1]])
		  if point[1] < max_right_x and point[1] > (max_right_x-50) and point[0] > max_right_y and point[0] < zero_line-5:
		    right_corrector.append([point[0], point[1]])
		# Determine the residuum of a linear fit along the contact edge
		if len(left_corrector) != 0:
		  left_fit = polyfit([x[1] for x in left_corrector], [x[0] for x in left_corrector], 1, full=True)
		  if debug:
		    print("left_fit: "+str(left_fit)+"\n")
		    if len(left_fit[1]) !=0: print("Res.: "+str((left_fit[1])[0])+"\n")
		    else: print("Res.: 0.0\n")
		  # res over 1000 for angles over 90 deg, res below 5 for angles close to 90 deg
		  if len(left_fit[1]) == 0 or (left_fit[1])[0] > 1000:
		    over90_left = True
		    fit_points_left = 15
		    for p, point in enumerate(left_corrector):
		      if point[1] > min_right_x:
		      	min_right_x = point[1]
		      	min_right_y = point[0]
		    left_point += np.argmax([x[1] for x in left_corrector])
		# the same for the right side
		if len(right_corrector) != 0:
		  right_fit = polyfit([x[1] for x in right_corrector], [x[0] for x in right_corrector], 1, full=True)
		  if debug:
		    print("right_fit: "+str(right_fit)+"\n")
		    if len(right_fit[1]) !=0: print("Res.: "+str((right_fit[1])[0])+"\n")
		    else: print("Res.: 0.0\n")
		  if len(right_fit[1]) == 0 or (right_fit[1])[0] > 1000:
		    over90_right = True
		    fit_points_right = 15
		    for p, point in enumerate(right_corrector):
		      if point[1] < max_right_x:
		      	max_right_x = point[1]
		      	max_right_y = point[0]
		    right_point -= len(right_corrector)-np.argmin([x[1] for x in right_corrector])
		  if debug:
		    print("New left point: "+str(rot_contours[left_point])+"\n")
		    print("New right point: "+str(rot_contours[right_point])+"\n")
		if (over90_left == True or over90_right == True) and reset90 != True:
		  reset90 = True
		  break
		tangent_x_right_orig = np.zeros(fit_points_right)
		tangent_y_right_orig = np.zeros(fit_points_right)
		tangent_x_left_orig = np.zeros(fit_points_left)
		tangent_y_left_orig = np.zeros(fit_points_left)
		
	
		# Calculate the width of the droplet
		width=math.sqrt(abs(max_right_x-min_right_x)**2 + abs(max_right_y-min_right_y)**2)
		
		# Grab points for the tangent fit
		for i in range(0, fit_points_right): 
			tangent_x_right_orig[i] = ((rot_contours)[right_point+i])[1]		
			tangent_y_right_orig[i] = ((rot_contours)[right_point+i])[0]
			right_dist_x = abs(((rot_contours)[right_point+i])[1]-((rot_contours)[right_point+i+1])[1])
			right_dist_y = abs(((rot_contours)[right_point+i])[0]-((rot_contours)[right_point+i+1])[0])
			right_dist = math.sqrt(right_dist_x**2+right_dist_y**2)
			if right_dist > 2:
			  for j in range(1, math.floor(right_dist)):
			    tangent_x_right_orig[i+j] = right_dist_x/math.floor(right_dist)
			    tangent_y_right_orig[i+j] = right_dist_y/math.floor(right_dist)
			    i += 1
		for i in range(0, fit_points_left): 
			tangent_x_left_orig[i] = ((rot_contours)[left_point-i])[1]		
			tangent_y_left_orig[i] = ((rot_contours)[left_point-i])[0]
			left_dist_x = abs(((rot_contours)[left_point+i])[1]-((rot_contours)[left_point+i+1])[1])
			left_dist_y = abs(((rot_contours)[left_point+i])[0]-((rot_contours)[left_point+i+1])[0])
			left_dist = math.sqrt(left_dist_x**2+left_dist_y**2)
			if left_dist > 2:
			  for j in range(1, math.floor(left_dist)):
			    tangent_x_left_orig[i+j] = left_dist_x/math.floor(left_dist)
			    tangent_y_left_orig[i+j] = left_dist_y/math.floor(left_dist)
			    i += 1
			
		# Filter the points for a smoother angle
		tangent_x_right = np.empty(0)
		tangent_y_right = np.empty(0)
		tangent_x_left = np.empty(0)
		tangent_y_left = np.empty(0)
		if over90_right == False:
		  while len(tangent_x_right_orig) > 0:
			  argmax = np.argmax(tangent_x_right_orig)
			  tangent_x_right = np.append(tangent_x_right, tangent_x_right_orig[argmax])		
			  tangent_y_right = np.append(tangent_y_right, tangent_y_right_orig[argmax])
			  tangent_x_right_orig = np.delete(tangent_x_right_orig, np.arange(argmax+1))
			  tangent_y_right_orig = np.delete(tangent_y_right_orig, np.arange(argmax+1))
		else:
		  tangent_x_right = tangent_x_right_orig
		  tangent_y_right = tangent_y_right_orig
		if over90_left == False:
		  while len(tangent_x_left_orig) > 0:
			  argmin = np.argmin(tangent_x_left_orig)
			  tangent_x_left = np.append(tangent_x_left, tangent_x_left_orig[argmin])		
			  tangent_y_left = np.append(tangent_y_left, tangent_y_left_orig[argmin])
			  tangent_x_left_orig = np.delete(tangent_x_left_orig, np.arange(argmin+1))	
			  tangent_y_left_orig = np.delete(tangent_y_left_orig, np.arange(argmin+1))		
		else:
		  tangent_x_left = tangent_x_left_orig
		  tangent_y_left = tangent_y_left_orig
		len_right = len(tangent_x_right)
		len_left = len(tangent_x_left)
		
		
		# Do a 2nd-degree polynomial fit
		fit2 = polyfit(tangent_x_right, tangent_y_right, 2)
		inf_point_x = tangent_x_right[0]
		inf_point_deltax = np.abs(tangent_x_right[0]-tangent_x_right[len_right-1])/(fit_points_right*10)
		
		# Change he distance between the used points
		for p, point in enumerate(tangent_x_right):
			tangent_x_right[p] = inf_point_x-p*inf_point_deltax 
			
		# Taylor the poly2-fit and do a poly1-fit
		taylor_right = fit2[0]*(inf_point_x)**2+fit2[1]*(inf_point_x)+fit2[2]+ (2*fit2[0]*inf_point_x+fit2[1])*(tangent_x_right-inf_point_x)
		fit1 = polyfit(tangent_x_right, taylor_right, 1)
		theta_right = math.degrees(abs(math.atan(fit1[0])))
		if over90_right == True: theta_right = 180-theta_right
		
		# Adjust the arrays for a more visible plot
		tangent_x_right = np.arange(tangent_x_right[len(tangent_x_right)-1]-width/3, tangent_x_right[0])
		taylor_right = fit1[0]*tangent_x_right+fit1[1]
		
		# Now do the same again for the left inflection point
		fit2 = polyfit(tangent_x_left, tangent_y_left, 2)
		inf_point_x = tangent_x_left[0]
		inf_point_deltax = np.abs(tangent_x_left[0]-tangent_x_left[len_left-1])/(fit_points_left*10) 
		for p, point in enumerate(tangent_x_left):
			tangent_x_left[p] = inf_point_x+p*inf_point_deltax
		taylor_left = fit2[0]*(inf_point_x)**2+fit2[1]*(inf_point_x)+fit2[2]+ (2*fit2[0]*inf_point_x+fit2[1])*(tangent_x_left-inf_point_x)
		fit1 = polyfit(tangent_x_left, taylor_left, 1)
		theta_left = math.degrees(abs(math.atan(fit1[0])))
		if over90_left == True: theta_left = 180-theta_left
		tangent_x_left = np.arange(tangent_x_left[0], tangent_x_left[len(tangent_x_left)-1]+width/3)
		taylor_left = fit1[0]*tangent_x_left+fit1[1]
		
	
	
		if theta_right==-1 or theta_left==-1: continue    # Skip file if the contact angles couldn't be found
		if debug:
			print("theta_right = "+str(theta_right)+"\n")
			print("theta_left = "+str(theta_left)+"\n")
		inflection_line_x = [min_right_x, max_right_x]
		inflection_line_y = [min_right_y, max_right_y]
		
	
		
		# Display the image and plot all contours and lines used for fitting
		if debug:
			fig, ax = plt.subplots()
			ax.imshow(imOrig, interpolation='nearest', cmap=plt.cm.gray)		
			ax.plot([x[1] for x in rot_contours], [x[0] for x in rot_contours], linewidth=1)
			for n, contour in enumerate(contours):
			    ax.plot(contour[:, 1], contour[:, 0], linewidth=1)
			ax.plot(tangent_x_left, taylor_left, linewidth=2)
			ax.plot(tangent_x_right, taylor_right, linewidth=2)
			ax.plot(surface_x, surface_y, linewidth=1)
			ax.plot(inflection_line_x, inflection_line_y, linewidth=1)	
			ax.axis('image')
			ax.set_xticks([])
			ax.set_yticks([])	
			plt.show()
		
		
		# Calculation of droplet velocity and travelled distance
		if filenumber > 1:
		  first_square = im[0:100] # the reference square for the bottom left corner
		  dist_left = min_right_x - first_left
		  dist_right = max_right_x - first_right
		  v_left = (min_right_x - pre_left)/shutter_time
		  v_right = (max_right_x - pre_right)/shutter_time
		else:
		  first_left = min_right_x
		  first_right = max_right_x
		  dist_left = 0
		  dist_right = 0
		pre_left = min_right_x
		pre_right = max_right_x
		
		reset90 = False
		filedif += 1
		
		for p, point in enumerate(rot_contours):
			XAchse.append(point[1])
			YAchse.append(point[0])

		a = len(XAchse)
		
		for i in range(right_point):
			del(XAchse[0])
			del(YAchse[0])
			
		for i in range(right_point):
			del(XAchse[-1])
			del(YAchse[-1])
			
		for i in XAchse:
			XAchse_gesamt.append(i)
		for i in YAchse:
			YAchse_gesamt.append(i*-1)
			
		Zeit.append(bb)
		bb = shutter_time+bb
		
		for i in range(len(XAchse)-1):
			Breite_infinitesimal = XAchse[i+1]-XAchse[i]
			Hoehe_infinitesimal = ((YAchse[i+1]-YAchse[0])+(YAchse[i]-YAchse[0]))/2.0
			Flaeche_infinitesimal = Breite_infinitesimal*Hoehe_infinitesimal
			A_i.append(Flaeche_infinitesimal)
			Summe_A = Summe_A+Flaeche_infinitesimal
			
		for i in range(len(XAchse)-1):
			Hoehe_Tropfen = YAchse[i+2]
			if Hoehe_Tropfen<=YAchse[i+1] and Hoehe_Tropfen<=YAchse[i] and Hoehe_Tropfen<=YAchse[i+3] and Hoehe_Tropfen<=YAchse[i+4]:
				Hoehe_Tropfen = YAchse[0]-Hoehe_Tropfen
				Tropfenhoehe.append(Hoehe_Tropfen)
				break
		for i in range(len(XAchse)-1):
			Schwerpunkt_x_i = 0.5*(XAchse[i+1]+XAchse[i])
			Schwerpunkt_y_i = 0.5*(((YAchse[i+1]-YAchse[0])+(YAchse[i]-YAchse[0]))/2.0)
			x_i.append(Schwerpunkt_x_i)
			y_i.append(Schwerpunkt_y_i)
			
		for s in range(len(A_i)):
			Summe_A_x = Summe_A_x + x_i[s]*A_i[s]
			Summe_A_y = Summe_A_y + y_i[s]*A_i[s]
			
		Schwerpunkt_x_s = Summe_A_x/Summe_A
		Schwerpunkt_y_s = Summe_A_y/Summe_A
		Schwerpunkt_y_s = (Schwerpunkt_y_s)+ YAchse[0]
		x_s.append(Schwerpunkt_x_s)
		y_s.append(Schwerpunkt_y_s)
		
		for i in range(len(XAchse)):
			del(XAchse[0])
		for i in range(len(YAchse)):
			del(YAchse[0])
			
		Summe_A=0
		Summe_A_x=0
		Summe_A_y=0
		x_i = []
		y_i = []
		A_i = []
		
		plot_file.write(str(file)+"\t"+str(filenumber)+"\t"+"{0:.3f}".format(dist_left)+"\t"+"{0:.3f}".format(dist_right)+"\t"+"{0:.3f}".format(v_left)+"\t"+"{0:.3f}".format(v_right)+"\t"+"{0:.3f}".format(alpha_surf)+"\t"+"{0:.3f}".format(width)+"\t"+"{0:.3f}".format(theta_left)+"\t"+"{0:.3f}".format(theta_right)+"\t"+"{0:.3f}".format(Schwerpunkt_x_s)+"\t"+"{0:.3f}".format(Schwerpunkt_y_s)+"\t"+"{0:.3f}".format(Hoehe_Tropfen)+"\n")
		                
# close file
plot_file.close()

