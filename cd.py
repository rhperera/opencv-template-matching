import cv2
import numpy as np

fileName = input("Enter image name: ")
cardClass = ""

t_hearts    = {"l":"h","t":cv2.imread('CardsPng/hearts.jpg',0)}
t_hearts2   = {"l":"h","t":cv2.imread('CardsPng/hearts2.jpg',0)}
t_spades    = {"l":"s","t":cv2.imread('CardsPng/spades.jpg',0)}
t_spades2   = {"l":"s","t":cv2.imread('CardsPng/spades2.jpg',0)}
t_clubs     = {"l":"c","t":cv2.imread('CardsPng/clubs.jpg',0)}
t_clubs2    = {"l":"c","t":cv2.imread('CardsPng/clubs2.jpg',0)}
t_diamonds  = {"l":"d","t":cv2.imread('CardsPng/diamonds.jpg',0)}
t_diamonds2 = {"l":"d","t":cv2.imread('CardsPng/diamonds2.jpg',0)}

test_image                  = cv2.imread(fileName)
test_image_gray             = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image_gray_blur        = cv2.GaussianBlur(test_image_gray,(5,5),0)
test_image_gray_laplassian  = cv2.Laplacian(test_image_gray_blur,cv2.CV_8U)
test_image_sharpened        = cv2.addWeighted(test_image_gray,0.7,test_image_gray_laplassian,0.3,0)

templateArray = [t_hearts,t_hearts2,t_spades,t_spades2,
                t_clubs,t_clubs2,t_diamonds,t_diamonds2]

for t in templateArray:
    temp_blur       = cv2.GaussianBlur(t['t'],(5,5),0)
    temp_laplassian = cv2.Laplacian(temp_blur,cv2.CV_8U)
    temp_sharpened  = cv2.addWeighted(t['t'],0.7,temp_laplassian,0.3,0)
    results = cv2.matchTemplate(test_image_sharpened,temp_sharpened, cv2.TM_CCOEFF_NORMED)
    locations = np.where(results >= 0.95)
    if len(locations[0])>=1:
        w, h = t['t'].shape[::-1]
        cardClass=t['l']
        for pt in zip(*locations[::-1]):
            cv2.rectangle(test_image, pt, (pt[0]+w, pt[1]+h),(0,255,255),2)
        print(cardClass)
        break

if cardClass=="h":
    print("Hearts")
elif cardClass=="s":
    print("Spades")
elif cardClass=="c":
    print("Clubs")
elif cardClass=="d":
    print("Diamonds")
else:
    print("Unable to detect")

cv2.imshow('detected',test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
