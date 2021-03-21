# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 20:43:59 2021

@author: Milo
"""

import os
import random
from PIL import Image

'''
MAKE SURE TO CHANGE THESE DIRECTORIES TO YOURS:
    cdir is the "Current Directory", its the dir you want to scan
    sdir is the "Solved Director", its the dir you want to save the images in
'''
# cdir = r'C:\VISION\flickr-scrape\images'
# sdir = r'C:\VISION\flickr-scrape\images\sorted'
cdir = r'C:\VISION\flickr-scrape\images'
sdir = r'C:\VISION\flickr-scrape\images\sorted'

def move(filename, name):
    try:
        os.rename(filename, sdir+"\\"+name)    
    except FileExistsError:
        outName = sdir+"\\"+str(random.randint(0,100))+name
        os.rename(filename, outName)
        print("Changed Image Name of: " + outName)

dirs = []
for root, subdirectories, files in os.walk(cdir):
    for subdirectory in subdirectories:
        path = os.path.join(root, subdirectory)
        if(path == sdir):
            continue
        dirs.append(path)
        print(path)

total   = 0
resized = 0
for subdir in dirs:
    #Go through all sub dirs
    for file in os.listdir(subdir):
        #In every subdir go through every file
        name = os.fsdecode(file)
        filename = subdir + str("\\") + name
        if( filename.endswith(".jpg")):
            #If the file is a jpg check rsolution and
            #act accoringly
            try:
                im = Image.open(filename)
            except:
                print("Couldn't open file: " + str(name))
                continue

            if((im.size[0] * 2) < (im.size[1] * 3) + 15 and(im.size[0] * 2) > (im.size[1] * 3) - 15 ): 
                total+=1
            elif((im.size[1] * 2) < (im.size[0] * 3) + 15 and (im.size[1] * 2) > (im.size[0] * 3) - 15):
                print("Rotated: " + str(name))
                im = im.rotate(90, expand=True)
                im.save(filename)
                total+=1
            else:
                if(im.size[0] < 768 or im.size[1] < 768):
                    print(str(name) + " is to small.")
                im.close()
                os.remove(filename)
                continue
            
            if(im.size[0] > 4096):
                print(str(name) + " is to big, resizing...")
                
                width = random.randint(2048, 4096)
                im = im.resize((width, int(width/1.5)))      
                im.save(filename)
                resized += 1

            total += 1
            im.close()
            move(filename, name)

        if( filename.endswith(".png")):
            name = os.fsdecode(file)
            filename = subdir + str("\\") + name
            os.remove(filename)
            print("Removed PNG image: " + name)

print("Total Resized images: " + str(resized))        
print("Total usable images: "  + str(total))


