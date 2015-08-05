#!/usr/bin/python
usage = "segbuilder.py [--options] config.ini"
description = "a script that lives persistently and build segments periodically"
author = "R. Essick (reed.essick@ligo.org)"

#import lal
from lal import gpstime
from pylal import Fr

import os
import glob
import sys ### only needed if we touch sys.stdout?
import multiprocessing as mp
import subprocess as sp
import numpy as np
import time
import logging
from ConfigParser import SafeConfigParser

from optparse import OptionParser


#=================================================
def report(statement, verbose):
	"""
	wrapper for reporting output
	"""
	if verbose:
		print statement
	logger.info(statement)

###	
def ldr_find_frames(ldr_server, ldr_url_type, ldr_type, ifo, start, stride, verbose=False):
	"""
	wrapper for ligo_data_find
	"""
	end = start+stride

	if ldr_server:
	        cmd = "ligo_data_find --server=%s --url-type=%s --type=%s --observatory=%s --gps-start-time=%d --gps-end-time=%d"%(ldr_server, ldr_url_type, ldr_type, ifo, start, end)
	else:
		cmd = "ligo_data_find --url-type=%s --type=%s --observatory=%s --gps-start-time=%d --gps-end-time=%d"%(ldr_url_type, ldr_type, ifo, start, end)
        report(cmd, verbose)

	p = sp.Popen(cmd.split(), stdout=sp.PIPE, stderr=sp.STDOUT)
	
	frames = p.communicate()[0].replace("No files found!", "").replace("\n", " ") ### handle an empty list appropriately

	frames = frames.replace("file://localhost","")

	return [l for l in frames.split() if l.endswith(".gwf")]

###
def shm_find_frames(directory, ifo, ldr_type, start, stride, verbose=False):
	"""
	searches for frames in directory assuming standard naming conventions.
	Does this by hand rather than relying on ldr tools
	"""
	end = start+stride

	frames = []
	for frame, (s, d) in [ (frame, extract_start_dur(frame, suffix=".gwf")) for frame in \
	                         sorted(glob.glob("%s/%s1/%s-%s-*-*.gwf"%(directory, ifo, ifo, ldr_type))) ]:
		if (s <= end) and (s+d >= start): ### there is some overlap!
			frames.append( frame )
	
	return frames

###
def extract_start_dur(filename, suffix=".gwf"):
	return [int(l) for l in filename[:-len(suffix)].split("-")[-2:]]

###
def coverage(frames, start, stride):
	"""
	determines the how much of [start, start+stride] is covered by these frames

	assumes non-overlapping frames!
	"""
	### generate segments from frame names
	segs = [[float(l) for l in frame.strip(".gwf").split("-")[-2:]] for frame in sorted(frames)]
	segs = [extract_start_dur(frame) for frame in sorted(frames)]

	### check whether segments overlap with desired time range
	covered = 1.0*stride

	end = start + stride
	for s, d in segs:
		e = s+d

		if (s < end) and (start < e): ### at least some overlap
			covered -= min(e, end) - max(s, start) ### subtract the overlap

		if covered <= 0:
			break

	return 1 - covered/stride ### return fraction of coverage

###
def extract_scisegs(frames, channel, bitmask, start, stride):
	"""
	extract scisegs from channel in frames using bitmask
	"""
	if not frames: ### empty list, so no segments
		return []

	### extract vectors and build segments
	segset = []
	for frame in frames:
		### extract the vector from the frame
		vect, s, ds, dt, xunit, yunit = Fr.frgetvect1d(frame, channel)		
		n = len(vect)

		### build time vector        add starting time
		t = np.arange(0, dt*n, dt) + s+ds

		### determine whether state acceptable
		### add "False" buffers to set up the computation of start and end time
#		state = np.concatenate( ([False], vect == bitmask, [False])) ### exact integer match
		state = np.concatenate( ([False], (vect >> bitmask) & 1, [False])) ### bitwise operation

		### determine beginning of segments
		###      i=False      i+1 = True  strip the trailing buffer
		b = ( (1-state[:-1])*(state[1:]) )[:-1].astype(bool)
		b = t[b] ### select out times

		### determine end of segments
                ###     i=True     i+1=False      strip the leading buffer
		e = ( (state[:-1])*(1-state[1:]) )[1:].astype(bool) 
		e = t[e] + dt ### select out times
		              ### extra dt moves these markers to the end of segments

		### stitch together start and end times, append to global list
		segset += list( np.transpose( np.array( [b, e] ) ) )

	if not segset: ### empty list
		return []

	### clean up segs!
	segs = []
	seg1 = segset[0]
	for seg2 in segset[1:]:
		if seg1[1] == seg2[0]:
			seg1[1] = seg2[1] ### join the segments
		else:
			### check segment for sanity
			append_seg = check_seg( seg1, (start, start+stride) )
			if append_seg:
				segs.append( append_seg )
			seg1 = seg2
	append_seg = check_seg( seg1, (start, start+stride) )
	if append_seg:
    		segs.append( append_seg )

	### return final list of lists!
	return segs

###
def check_seg(seg, window=None):
	"""
	ensures segment makes sense
	if provided, segment is sliced so that we only keep the part that interesects with window=[start, end]
	"""
	s, e = seg
	if window:
		start, end = window
		if s < start:
			s = start # truncate the start of seg
		elif not (s < end):
			return False # no overlap between current segment and window
		if e > end:
			e = end # truncate the end of seg
		elif not (e > start):
			return False # no overlap between currnet segment and window

	if s < e:
		return [s,e]
	elif s > e:
		raise ValueError("something is very wrong with segment generation... seg[1]=%.3f < seg[0]=%.3f"%tuple(seg))
	else:
		return False

#=================================================

parser = OptionParser(usage=usage, description=description)

parser.add_option("-v", "--verbose", default=False, action="store_true")
parser.add_option("-s", "--gps-start", default=None, type="int")
parser.add_option("-e", "--gps-end", default=np.infty, type="float")

parser.add_option("", "--shared-mem", default=False, action="store_true", help="find frames in the shared memory partition rather than through ligo_data_find")

opts, args = parser.parse_args()

if len(args) != 1:
	raise StandardError("Please supply only a single argument")
configfile = args[0]

#=================================================

config = SafeConfigParser()
config.read(configfile)

#=================================================
### setup logger to record processes
logfilename = config.get("general","logfile")

### ensure that path to log will exist
logpath = "/".join(logfilename.split("/")[:-1])
if logfilename[0] == "/":
	logpath = "/%s"%logpath
if not os.path.exists(logpath):
	os.makedirs(logpath)

global logger
logger = logging.getLogger('crawler_log')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(message)s')

### redirect stderr into logger
hdlr = logging.FileHandler(logfilename)
hdlr.setFormatter(formatter)
hdlr.setLevel(logging.INFO)
logger.addHandler(hdlr)

#=================================================
report("pulling out parameters from : %s"%configfile, opts.verbose)

### pull out basic parameters
ifo = config.get("general", "ifo")
outputdir = config.get("general", "outputdir")
stride = config.getint("general", "stride")
delay = config.getint("general", "delay")
padding = config.getint("general", "padding")
twopadding = 2*padding

max_wait = config.getint("general", "max_wait")

#========================

### output formatting







raise StandardError("WRITE config.ini OPTIONS FOR OUTPUT FORMAT")







#========================

if opts.shared_mem:
	shmdir = config.get('shared-mem', 'directory')
	ldr_type = config.get('shared-mem', 'type')
else:
	### pull out ldr params
	if config.has_option("ligo_data_find", "server"):
		ldr_server = config.get("ligo_data_find", "server")
	else:
		ldr_server = None
	ldr_url_type = config.get("ligo_data_find", "url-type")
	ldr_type = config.get("ligo_data_find", "type")

#========================

### pull out sciseg params
sciseg_channel = config.get("scisegs","channel")
sciseg_bitmask = config.getint("scisegs","bitmask")

#=================================================

### setting up initial time
report("", opts.verbose)
if opts.gps_start == None:
	t = ( int(gpstime.gps_time_now()) /stride)*stride
else:
	t = (opts.gps_start/stride)*stride ### round to integer number of strides


























#=================================================
# LOOP until we "finish"
#=================================================
while t < opts.gps_end:

	report("=========================================================================", opts.verbose)
	report("processing stride: [%d-%d, %d+%d]"%(t, padding, t+stride, padding), opts.verbose)

	### wait to analyze this stride
	nowgps = float( gpstime.gps_time_now() )
	wait = (t+duration+padding) + delay - nowgps 
	if wait > 0:
		report("sleeping for %d sec"%wait, opts.verbose)
		time.sleep(wait)

	### build directories
	t5 = t/100000
	segdir = "%s/segments/%s-%d/"%(outputdir, ifo, t5)
	framedir = "%s/frames/%s-%d/"%(outputdir, ifo, t5)
	for directory in [segdir, framedir]:
        	if not os.path.exists(directory):
			report("building directory : %s"%directory, opts.verbose)
                	os.makedirs(directory)

	### find frames within time window
	report("finding frames within stride", opts.verbose)
#	frames = find_frames(ldr_server, ldr_url_type, ldr_type, ifo, t-padding, stride+2*padding, verbose=opts.verbose) 
#	covered = coverage( frames, t-padding, stride+2*padding) ### find out the coverage
	if opts.shared_mem:
		frames = shm_find_frames(shmdir, ifo, ldr_type, t-padding, duration+twopadding, verbose=opts.verbose)
	else:
		frames = ldr_find_frames(ldr_server, ldr_url_type, ldr_type, ifo, t-padding, duration+twopadding, verbose=opts.verbose) 

	covered = coverage( frames, t-padding, duration+2*padding) ### find out the coverage

	### keep looking every second until we either find frames or time out
	if covered < 1.0:
		report("coverage = %.5f < 1.0, we'll check every second for more frames and wait at most %d seconds before proceeding."%(covered, max_wait), opts.verbose)

#	while (covered < 1.0) and ( (float(gpstime.gps_time_now()) - ( (t+stride+padding) + delay ) ) < max_wait ):
	while (covered < 1.0) and ( (float(gpstime.gps_time_now()) - ( (t+duration+twopadding) + delay ) ) < max_wait ):

		###
		time.sleep( 1 ) # don't break the file system
		###

#		frames = find_frames(ldr_server, ldr_url_type, ldr_type, ifo, t-padding, stride+2*padding, verbose=False) ### don't report this every time in the loop
#		covered = coverage( frames, t-padding, stride+2*padding) ### find out the coverage
		if opts.shared_mem:
			frames = shm_find_frames(shmdir, ifo, ldr_type, t-padding, duration+twopadding, verbose=opts.verbose)
		else:
			frames = ldr_find_frames(ldr_server, ldr_url_type, ldr_type, ifo, t-padding, duration+twopadding, verbose=False) ### don't report this every time in the loop
		covered = coverage( frames, t-padding, duration+twopadding) ### find out the coverage

		if covered >= 1.0:
			report("covered >= 1.0", opts.verbose)

	if covered < 1.0:
		report("coverage = %.5f < 1.0, but we've timed out after waiting at least %d seconds."%(covered, max_wait), opts.verbose)

	### write framecache
#	framecache = "%s/%s_%d-%d.lcf"%(framedir, ifo, t, stride)
	framecache = "%s/%s_%d-%d.lcf"%(framedir, ifo, t-padding, duration+twopadding)
	report("writing framecache : %s"%framecache, opts.verbose)

	framecache_obj = open(framecache, "w")
	framecache_obj.write( str_framecache(frames, ifo, ldr_type) )
	framecache_obj.close()

	### if we have data, process it!
	if not frames:
		report("no frames found! skipping...", opts.verbose)

	else:
		### find scisegs
#		segfile = "%s/%s_%d-%d.seg"%(segdir, ifo, t, stride)
		segfile = "%s/%s_%d-%d.seg"%(segdir, ifo, t-padding, duration+twopadding)
		report("extracting scisegs to : %s"%(segfile), opts.verbose)
#		segs = extract_scisegs(frames, "%s1:%s"%(ifo, sciseg_channel), sciseg_bitmask, t-padding, stride+twopadding)
		segs = extract_scisegs(frames, "%s1:%s"%(ifo, sciseg_channel), sciseg_bitmask, t-padding, duration+twopadding)

		report("writing scisegs : %s"%segfile, opts.verbose)
		file_obj = open(segfile, "w")
		for a, b in segs:
			file_obj.write("%d %d"%(a, b))
		file_obj.close()

	report("Done with stride: [%d-%d, %d+%d]"%(t, padding, t+stride, padding), opts.verbose)

	### increment!
	t += stride
