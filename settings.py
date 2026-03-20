##########################################################################################################
#                                                                                                        #
#  $$$$$$$\  $$\                            $$\                                    $$$$$$\  $$\    $$\   #
#  $$  __$$\ $$ |                           $$ |                                  $$  __$$\ $$ |   $$ |  #
#  $$ |  $$ |$$$$$$$\   $$$$$$\  $$$$$$$\ $$$$$$\    $$$$$$\  $$$$$$\$$$$\        $$ /  \__|$$ |   $$ |  #
#  $$$$$$$  |$$  __$$\  \____$$\ $$  __$$\\_$$  _|  $$  __$$\ $$  _$$  _$$\       $$ |      \$$\  $$  |  #
#  $$  ____/ $$ |  $$ | $$$$$$$ |$$ |  $$ | $$ |    $$ /  $$ |$$ / $$ / $$ |      $$ |       \$$\$$  /   #
#  $$ |      $$ |  $$ |$$  __$$ |$$ |  $$ | $$ |$$\ $$ |  $$ |$$ | $$ | $$ |      $$ |  $$\   \$$$  /    #
#  $$ |      $$ |  $$ |\$$$$$$$ |$$ |  $$ | \$$$$  |\$$$$$$  |$$ | $$ | $$ |      \$$$$$$  |   \$  /     #
#  \__|      \__|  \__| \_______|\__|  \__|  \____/  \______/ \__| \__| \__|       \______/     \_/      #
#                                                                                                        #
##########################################################################################################
#                                                                                                        #
#                                  Hello, Welcome to Phantom CV!                                         #
#                                                                                                        #
#                Please consider donating, this took a lot of research and development                   #
#                                                                                                        #
#                      BIG thanks to YOLOv12 by Ultralytics and the community                           #
#                                                                                                        #
##########################################################################################################
#                                                                                                        #
# Bitcoin - 1Fwdk5fkhZ1Y3uyb2HKywHyHQuddiik6Xu     Ethereum - 0x794067Aa418A6bBe774eE313140cCDFd35C2a148 #
#                                                                                                        #
# Dogecoin - DS8q14xNtRD3FZAzpmTrGfJ88Ain61wXor    Ripple - rMVLwS4A5onNvcCPYAeqS2fq3DQg6DRr7L           #
#                                                                                                        #
##########################################################################################################
###     GENERAL SETTINGS    ###
dataCollectionMode = False  # Captures training data as you play.
showBoundingBox = True  # Enable / Disable the drawing of bounding boxes on the screen

###     AIM ASSIST SETTINGS    ###
aimAssist = True  # Enable / Disable Advanced Aim Assist
autoFireDistance = 0  # How close (in pixels) before auto-fire triggers. Set to 0 to disable.
confidence = 0.6  # Confidence threshold for predictions (0-1).
IoUThreshold = 0.45  # Intersection over Union threshold (0-1).
speedX, speedY = 9, 9  # Speed modifier for aim on X and Y axes.
aimSmoothing = False  # Pull harder towards targets closer to center
boundingBoxX1, boundingBoxY1, boundingBoxX2, boundingBoxY2 = 0, 0, 1920, 1080  # Screen region to scan.
boundingBoxColor = [102, 51, 153]  # RGB color for bounding boxes
adaptiveBoundingBox = False  # Automatically resize bounding box when aiming in
manualOveride = True  # Manually override model inputs when pressing left stick.
modelName = 'warzone'  # Name of the YOLOv12 .pt model file to use (without extension).
