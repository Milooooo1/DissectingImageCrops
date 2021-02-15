# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 20:43:59 2021

@author: Milo
"""

import os
from PIL import Image


def move(cdir):
    try:
        os.rename(filename, sdir+"\\"+name)    
    except:
        print("Image in use.")

cdir = r'C:\VISION\flickr-scrape\images'
sdir = r'C:\VISION\flickr-scrape\images\sorted'

dirs = []
for root, subdirectories, files in os.walk(cdir):
    for subdirectory in subdirectories:
        path = os.path.join(root, subdirectory)
        if(path == sdir):
            continue
        dirs.append(path)
        print(path)

total = 0
for subdir in dirs:
    for file in os.listdir(subdir):
        name = os.fsdecode(file)
        filename = subdir + str("\\") + name
        if( filename.endswith(".jpg")):
            im = Image.open(filename)
            # print(im.size)
            if((im.size[0] * 2) < (im.size[1] * 3) + 15 and(im.size[0] * 2) > (im.size[1] * 3) - 15 ): 
                print(filename) 
                total+=1
                im.close()
                move(filename)
            elif((im.size[1] * 2) < (im.size[0] * 3) + 15 and (im.size[1] * 2) > (im.size[0] * 3) - 15):
                print("Rotated: " + str(filename))
                im = im.rotate(90, expand=True)
                im.save(filename)
                total+=1
                im.close()
                move(filename)
            else:
                im.close()
                os.remove(filename)
        
print("Total usable images: " + str(total))

