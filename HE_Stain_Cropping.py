'''Author: edong'''

# cropping and augmentation
from PIL import Image
import glob, os
from numpy import median

# Import Pillow:
# http://matthiaseisen.com/pp/patterns/p0202/

savepath = "K:\\edong\\MedBrain\\21_Crops_on_Resized"
readpath = "K:\\edong\\MedBrain\\NormalizeDataSet"


# one example
# os.chdir("K:\\edong\\MedBrain\\NormalizeDataSet\\onenormalized")
# img = Image.open('107_005A_H&E_A-3-3.tif')
# width, height = img.size


# Get the median value for image dimensions:
def getImgMedianDimension(path):
    listdir = os.listdir(path)
    wlist = []
    hlist = []
    for dir in listdir:
        os.chdir(readpath + '/' + dir)
        print('Loaded the images of dataset-' + '{}\n'.format(dir))
        listimgs = glob.glob('*.tif')
        for i in listimgs:
            file, ext = os.path.splitext(i)
            # Load the original image:
            img = Image.open(i)
            width, height = img.size
            wlist.append(width)
            hlist.append(height)
    w = median(wlist)
    h = median(hlist)
    return w, h


medwidth, medheight = getImgMedianDimension(path=readpath) #2518, 2474 -> Therefore, resize to 2500*2500


# original and resized centres.
circle_diameter = 2000  # pixels, so resized diametre is 1000, and if crop is 300, then it's 1/3 of the resized
# as indicated on my PPT literature summary conclusion



def list_centres(wcentre, hcentre):
    # List of original centres
    # list_of_orcentres = [[wcentre - 300, hcentre - 600],  # 1
    #                      [wcentre, hcentre - 600],  # 2
    #                      [wcentre + 300, hcentre - 600],  # 3
    #
    #                      [wcentre - 600, hcentre - 300],  # 4
    #                      [wcentre - 300, hcentre - 300],  # 5
    #                      [wcentre, hcentre - 300],  # 6
    #                      [wcentre + 300, hcentre - 300],  # 7
    #                      [wcentre + 600, hcentre - 300],  # 8
    #
    #                      [wcentre - 600, hcentre],  # 9
    #                      [wcentre - 300, hcentre],  # 10
    #                      [wcentre, hcentre],  # 11
    #                      [wcentre + 300, hcentre],  # 12
    #                      [wcentre + 600, hcentre],  # 13
    #
    #                      [wcentre - 600, hcentre + 300],  # 14
    #                      [wcentre - 300, hcentre + 300],  # 15
    #                      [wcentre, hcentre + 300],  # 16
    #                      [wcentre + 300, hcentre + 300],  # 17
    #                      [wcentre + 600, hcentre + 300],  # 18
    #
    #                      [wcentre - 300, hcentre + 600],  # 19
    #                      [wcentre, hcentre + 600],  # 20
    #                      [wcentre + 300, hcentre + 600],  # 21
    #                      ]

    # List of resized centres
    list_of_recentres = [[wcentre - 150, hcentre - 300],  # 1
                         [wcentre, hcentre - 300],  # 2
                         [wcentre + 150, hcentre - 300],  # 3

                         [wcentre - 300, hcentre - 150],  # 4
                         [wcentre - 150, hcentre - 150],  # 5
                         [wcentre, hcentre - 150],  # 6
                         [wcentre + 150, hcentre - 150],  # 7
                         [wcentre + 300, hcentre - 150],  # 8

                         [wcentre - 300, hcentre],  # 9
                         [wcentre - 150, hcentre],  # 10
                         [wcentre, hcentre],  # 11
                         [wcentre + 150, hcentre],  # 12
                         [wcentre + 300, hcentre],  # 13

                         [wcentre - 300, hcentre + 150],  # 14
                         [wcentre - 150, hcentre + 150],  # 15
                         [wcentre, hcentre + 150],  # 16
                         [wcentre + 150, hcentre + 150],  # 17
                         [wcentre + 300, hcentre + 150],  # 18

                         [wcentre - 150, hcentre + 300],  # 19
                         [wcentre, hcentre + 300],  # 20
                         [wcentre + 150, hcentre + 300],  # 21
                         ]

    return list_of_recentres


# New crop size: 300*300
def resizeAndCrop_NormalizedImg(resizewidth, resizeheight):
    listdir = os.listdir(readpath)
    for dir in listdir:

        if dir == 'onenormalized':
            folder = 'onecroped'
        else:
            folder = 'zerocroped'

        print('Loaded the images of dataset-' + '{}\n'.format(dir))

        os.chdir(readpath + '/' + dir)
        listimgs = glob.glob('*.tif')
        for i in listimgs:
            file, ext = os.path.splitext(i)
            # Load the original image:
            os.chdir(readpath + '/' + dir)
            imgori = Image.open(i)
            img = imgori.resize(resizewidth, resizeheight)

            # 2224px * 2224px, Starting in the center
            half_the_width = img.size[0] / 2
            half_the_height = img.size[1] / 2
            list_recentres = list_centres(half_the_width, half_the_height)

            ind = 0
            for i in list_recentres:
                imgsub = img.crop(
                    (i[0] - 150,
                     i[1] - 150,
                     i[0] + 150,
                     i[1] + 150)
                )
                ind += 1
                os.chdir(savepath + '/' + folder)
                imgsub.save(file + str(ind) + '.tif', "TIFF")

resizeAndCrop_NormalizedImg(int(0.5 * medwidth), int(0.5 * medheight))

            # imgcenter = img.crop(
            #     (
            #         half_the_width - 256,
            #         half_the_height - 256,
            #         half_the_width + 256,
            #         half_the_height + 256
            #     )
            # )
            #
            # imgrightdown = img.crop(
            #     (
            #         half_the_width + 256 - 256,
            #         half_the_height + 256 - 256,
            #         half_the_width + 256 + 256,
            #         half_the_height + 256 + 256
            #     )
            # )
            #
            # imgleftup = img.crop(
            #     (
            #         half_the_width - 256 - 256,
            #         half_the_height - 256 - 256,
            #         half_the_width - 256 + 256,
            #         half_the_height - 256 + 256
            #     )
            # )
            #
            # imgrightup = img.crop(
            #     (
            #         half_the_width + 256 - 256,
            #         half_the_height - 256 - 256,
            #         half_the_width + 256 + 256,
            #         half_the_height - 256 + 256
            #     )
            # )
            #
            # imgleftdown = img.crop(
            #     (
            #         half_the_width - 256 - 256,
            #         half_the_height + 256 - 256,
            #         half_the_width - 256 + 256,
            #         half_the_height + 256 + 256
            #     )
            # )

            # os.chdir(savepath + '/' + folder)
            # imgcenter.save(file + "c" + '.tif', "TIFF")
            # imgleftdown.save(file + "ld" + '.tif', "TIFF")
            # imgleftup.save(file + "lu" + '.tif', "TIFF")
            # imgrightdown.save(file + "rd" + '.tif', "TIFF")
            # imgrightup.save(file + "ru" + '.tif', "TIFF")

            # imgcenter.show()
            # imgleftdown.show()
            # imgleftup.show()
            # imgrightdown.show()
            # imgrightup.show()



