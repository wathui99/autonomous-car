import os
imgs = [im for im in os.scandir('.') if 'png' in im.name]
n=1
for img in imgs:
	os.rename(img.name, '{}.png'.format(n))
	n+=1
	#os.rename(old_file_name, new_file_name) 