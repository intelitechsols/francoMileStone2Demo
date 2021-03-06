import cv2
from numpy import math, hstack

import numpy as np

import cv2 as OOO0OOO0OOO0O0OO0 #line:1
from numpy import math as O00000O0O0000000O ,hstack as OO00O0O0O00OOO0OO #line:2
import numpy as OO0O00O000OOO0OOO #line:4
class FileVideoCapture (object ):#line:7
	def __init__ (OO0O0OOO0OO0OOOOO ,OO000OOO0O00OO000 ):#line:9
		OO0O0OOO0OO0OOOOO .path =OO000OOO0O00OO000 #line:10
		OO0O0OOO0OO0OOOOO .frame =1 #line:11
	def isOpened (O0OO0O000OO0OOOO0 ):#line:13
		O00000OO00O000OO0 =OOO0OOO0OOO0O0OO0 .imread (O0OO0O000OO0OOOO0 .path .format (O0OO0O000OO0OOOO0 .frame ))#line:14
		return O00000OO00O000OO0 !=None #line:15
	def read (OO00OOOO0OOOO0O0O ):#line:17
		OO00OO00O000O00OO =OOO0OOO0OOO0O0OO0 .imread (OO00OOOO0OOOO0O0O .path .format (OO00OOOO0OOOO0O0O .frame ))#line:18
		OO000OO00OOOO0OO0 =OO00OO00O000O00OO !=None #line:19
		if OO000OO00OOOO0OO0 :#line:20
			OO00OOOO0OOOO0O0O .frame +=1 #line:21
		return OO000OO00OOOO0OO0 ,OO00OO00O000O00OO #line:22
def squeeze_pts (O0OO0O0O00OOO0O00 ):#line:24
	O0OO0O0O00OOO0O00 =O0OO0O0O00OOO0O00 .squeeze ()#line:25
	if len (O0OO0O0O00OOO0O00 .shape )==1 :#line:26
		O0OO0O0O00OOO0O00 =OO0O00O000OOO0OOO .array ([O0OO0O0O00OOO0O00 ])#line:27
	return O0OO0O0O00OOO0O00 #line:28
def array_to_int_tuple (O0OOOO000OO00O0OO ):#line:30
	return (int (O0OOOO000OO00O0OO [0 ]),int (O0OOOO000OO00O0OO [1 ]))#line:31
def L2norm (OO000OOO00O00OOO0 ):#line:33
	return OO0O00O000OOO0OOO .sqrt ((OO000OOO00O00OOO0 **2 ).sum (axis =1 ))#line:34
current_pos =None #line:36
tl =None #line:37
br =None #line:38
def get_rect (O00O0O00000O000OO ,title ='get_rect'):#line:40
	global current_pos #line:42
	global tl #line:43
	global br #line:44
	global released_once #line:45
	current_pos =None #line:47
	tl =None #line:48
	br =None #line:49
	released_once =False #line:50
	OOO0OOO0OOO0O0OO0 .namedWindow (title )#line:52
	OOO0OOO0OOO0O0OO0 .moveWindow (title ,100 ,100 )#line:53
	def O0OO0O0OOOO00O000 (O000OO0000OOO0O00 ,O0OOO0O00OO00O0OO ,O0O0000000OO00000 ,O00O0O0000O0OO000 ,OOOO0OOO00OOOO000 ):#line:55
		global current_pos #line:56
		global tl #line:57
		global br #line:58
		global released_once #line:59
		current_pos =(O0OOO0O00OO00O0OO ,O0O0000000OO00000 )#line:61
		if tl is not None and not (O00O0O0000O0OO000 &OOO0OOO0OOO0O0OO0 .EVENT_FLAG_LBUTTON ):#line:63
			released_once =True #line:64
		if O00O0O0000O0OO000 &OOO0OOO0OOO0O0OO0 .EVENT_FLAG_LBUTTON :#line:66
			if tl is None :#line:67
				tl =current_pos #line:68
			elif released_once :#line:69
				br =current_pos #line:70
	OOO0OOO0OOO0O0OO0 .setMouseCallback (title ,O0OO0O0OOOO00O000 )#line:72
	OOO0OOO0OOO0O0OO0 .imshow (title ,O00O0O00000O000OO )#line:73
	while br is None :#line:75
		O0OO00OOO000O000O =OO0O00O000OOO0OOO .copy (O00O0O00000O000OO )#line:76
		if tl is not None :#line:78
			OOO0OOO0OOO0O0OO0 .rectangle (O0OO00OOO000O000O ,tl ,current_pos ,(255 ,0 ,0 ))#line:79
		OOO0OOO0OOO0O0OO0 .imshow (title ,O0OO00OOO000O000O )#line:81
		_O0OOO0OO0OOO00OO0 =OOO0OOO0OOO0O0OO0 .waitKey (10 )#line:82
	OOO0OOO0OOO0O0OO0 .destroyWindow (title )#line:84
	return (tl ,br )#line:86
def in_rect (OOO00O0000OOOOOO0 ,OOO00O0OO0O0000OO ,OO0O00OOO00OO000O ):#line:88
	if type (OOO00O0000OOOOOO0 )is list :#line:89
		OOO00O0000OOOOOO0 =keypoints_cv_to_np (OOO00O0000OOOOOO0 )#line:90
	O00O0OOOOO0000O00 =OOO00O0000OOOOOO0 [:,0 ]#line:92
	OOO000O0O0OO000O0 =OOO00O0000OOOOOO0 [:,1 ]#line:93
	O0O00O0O00O0OO0OO =O00O0OOOOO0000O00 >OOO00O0OO0O0000OO [0 ]#line:95
	O0O0OOO0000O00O00 =OOO000O0O0OO000O0 >OOO00O0OO0O0000OO [1 ]#line:96
	OO0O000O0O0OOOO00 =O00O0OOOOO0000O00 <OO0O00OOO00OO000O [0 ]#line:97
	OO0OO0O0O00O000O0 =OOO000O0O0OO000O0 <OO0O00OOO00OO000O [1 ]#line:98
	OOO0OO00O0OOO000O =O0O00O0O00O0OO0OO &O0O0OOO0000O00O00 &OO0O000O0O0OOOO00 &OO0OO0O0O00O000O0 #line:100
	return OOO0OO00O0OOO000O #line:102
def keypoints_cv_to_np (OO0OO00OOOOOOOOOO ):#line:104
	O0OO0O00O0OOO00OO =OO0O00O000OOO0OOO .array ([OOO00OOOOO00OOO00 .pt for OOO00OOOOO00OOO00 in OO0OO00OOOOOOOOOO ])#line:105
	return O0OO0O00O0OOO00OO #line:106
def find_nearest_keypoints (O0O0O00000000O0OO ,OOO00OOOOO0O000O0 ,number =1 ):#line:108
	if type (OOO00OOOOO0O000O0 )is tuple :#line:109
		OOO00OOOOO0O000O0 =OO0O00O000OOO0OOO .array (OOO00OOOOO0O000O0 )#line:110
	if type (O0O0O00000000O0OO )is list :#line:111
		O0O0O00000000O0OO =keypoints_cv_to_np (O0O0O00000000O0OO )#line:112
	OOOOOOO000O000000 =OO0O00O000OOO0OOO .sqrt (OO0O00O000OOO0OOO .power (O0O0O00000000O0OO -OOO00OOOOO0O000O0 ,2 ).sum (axis =1 ))#line:114
	OOO0O00O00000OO0O =OO0O00O000OOO0OOO .argsort (OOOOOOO000O000000 )#line:115
	return OOO0O00O00000OO0O [:number ]#line:116
def draw_keypoints (OOOO0OOOOOO00O0OO ,O00O00OO00OOO0OO0 ,color =(255 ,0 ,0 )):#line:118
	for OO000OOOO00OOO0OO in OOOO0OOOOOO00O0OO :#line:120
		OO000O00OOOO0O0O0 =3 #line:121
		OO00O00OO0O0OOO00 =(int (OO000OOOO00OOO0OO [0 ]),int (OO000OOOO00OOO0OO [1 ]))#line:122
		OOO0OOO0OOO0O0OO0 .circle (O00O00OO00OOO0OO0 ,OO00O00OO0O0OOO00 ,OO000O00OOOO0O0O0 ,color )#line:125
def track (OO0OO000O0O00O00O ,OOOOO00O0O00O00O0 ,O0O0OO000O0OOOOOO ,THR_FB =20 ):#line:127
	if type (O0O0OO000O0OOOOOO )is list :#line:128
		O0O0OO000O0OOOOOO =keypoints_cv_to_np (O0O0OO000O0OOOOOO )#line:129
	OOOOOO000OOO000O0 =O0O0OO000O0OOOOOO .shape [0 ]#line:131
	OO00OO00O00O00O00 =[False ]*OOOOOO000OOO000O0 #line:134
	if OOOOOO000OOO000O0 >0 :#line:137
		OOOOOO00OOO000OOO =O0O0OO000O0OOOOOO [:,None ,:2 ].astype (OO0O00O000OOO0OOO .float32 )#line:142
		OOOO00OO00O0OOO0O ,OO00OO00O00O00O00 ,_OO00000OOO0OOOOOO =OOO0OOO0OOO0O0OO0 .calcOpticalFlowPyrLK (OO0OO000O0O00O00O ,OOOOO00O0O00O00O0 ,OOOOOO00OOO000OOO ,None )#line:145
		O0O000O0O000O0O0O ,_OO00000OOO0OOOOOO ,_OO00000OOO0OOOOOO =OOO0OOO0OOO0O0OO0 .calcOpticalFlowPyrLK (OOOOO00O0O00O00O0 ,OO0OO000O0O00O00O ,OOOO00OO00O0OOO0O ,None )#line:148
		O0O000O0O000O0O0O =squeeze_pts (O0O000O0O000O0O0O )#line:151
		OOOOOO00OOO000OOO =squeeze_pts (OOOOOO00OOO000OOO )#line:152
		OOOO00OO00O0OOO0O =squeeze_pts (OOOO00OO00O0OOO0O )#line:153
		OO00OO00O00O00O00 =OO00OO00O00O00O00 .squeeze ()#line:154
		O00O0OO000O0OO00O =OO0O00O000OOO0OOO .sqrt (OO0O00O000OOO0OOO .power (O0O000O0O000O0O0O -OOOOOO00OOO000OOO ,2 ).sum (axis =1 ))#line:157
		O0OOOOO000000OO0O =O00O0OO000O0OO00O >THR_FB #line:160
		OO00OO00O00O00O00 =~O0OOOOO000000OO0O &OO00OO00O00O00O00 .astype (OO0O00O000OOO0OOO .bool )#line:161
		OOOO00OO00O0OOO0O =OOOO00OO00O0OOO0O [OO00OO00O00O00O00 ,:]#line:163
		OO000O0OO0OO000OO =O0O0OO000O0OOOOOO [OO00OO00O00O00O00 ,:]#line:164
		OO000O0OO0OO000OO [:,:2 ]=OOOO00OO00O0OOO0O #line:165
	else :#line:167
		OO000O0OO0OO000OO =OO0O00O000OOO0OOO .array ([])#line:168
	return OO000O0OO0OO000OO ,OO00OO00O00O00O00 #line:169
def rotate (OO000OOO00OOOOO0O ,OOOOO000O0OOOO0OO ):#line:171
	if (OOOOO000O0OOOO0OO ==0 ):#line:172
		return OO000OOO00OOOOO0O #line:173
	OO0000O000000OOOO =OO0O00O000OOO0OOO .empty (OO000OOO00OOOOO0O .shape )#line:175
	O0O0O0O0OOOOOOO0O ,O0OO0OO0O0O0O0OOO =[O0O0OOO00O0OOO0O0 (OOOOO000O0OOOO0OO )for O0O0OOO00O0OOO0O0 in (O00000O0O0000000O .sin ,O00000O0O0000000O .cos )]#line:177
	OO0000O000000OOOO [:,0 ]=O0OO0OO0O0O0O0OOO *OO000OOO00OOOOO0O [:,0 ]-O0O0O0O0OOOOOOO0O *OO000OOO00OOOOO0O [:,1 ]#line:179
	OO0000O000000OOOO [:,1 ]=O0O0O0O0OOOOOOO0O *OO000OOO00OOOOO0O [:,0 ]+O0OO0OO0O0O0O0OOO *OO000OOO00OOOOO0O [:,1 ]#line:180
	return OO0000O000000OOOO #line:182
def br (OOO0O0000OOO0O0O0 ):#line:184
	OOO0O00OOOOOO00OO =OO00O0O0O00OOO0OO ((OOO0O0000OOO0O0O0 [:,[0 ]]+OOO0O0000OOO0O0O0 [:,[2 ]]-1 ,OOO0O0000OOO0O0O0 [:,[1 ]]+OOO0O0000OOO0O0O0 [:,[3 ]]-1 ))#line:186
	return OOO0O00OOOOOO00OO #line:188
def bb2pts (O0O0000OOO00000OO ):#line:190
	O00OO00OOO00000OO =OO00O0O0O00OOO0OO ((O0O0000OOO00000OO [:,:2 ],br (O0O0000OOO00000OO )))#line:192
	return O00OO00OOO00000OO
#e9015584e6a44b14988f13e2298bcbf9