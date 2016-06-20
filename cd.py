import cv2
import numpy as np

fileName = input("Enter image name: ")
cardClass = ""

t_hearts  = {"l":"h","t":cv2.imread('CardsPng/hearts.jpg',0)}
t_hearts2  = {"l":"h","t":cv2.imread('CardsPng/hearts2.jpg',0)}
t_spades  = {"l":"s","t":cv2.imread('CardsPng/spades.jpg',0)}
t_spades2  = {"l":"s","t":cv2.imread('CardsPng/spades2.jpg',0)}
t_clubs  = {"l":"c","t":cv2.imread('CardsPng/clubs.jpg',0)}
t_clubs2  = {"l":"c","t":cv2.imread('CardsPng/clubs2.jpg',0)}
t_diamonds  = {"l":"d","t":cv2.imread('CardsPng/diamonds.jpg',0)}
t_diamonds2  = {"l":"d","t":cv2.imread('CardsPng/diamonds2.jpg',0)}

test1  = cv2.imread(fileName)
test1gray = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
templateArray = [t_hearts,t_hearts2,t_spades,t_spades2,
                t_clubs,t_clubs2,t_diamonds,t_diamonds2]

for t in templateArray:
    results = cv2.matchTemplate(test1gray,t['t'], cv2.TM_CCOEFF_NORMED)
    locations = np.where(results >= 0.95)
    if len(locations[0])>=1:
        w, h = t['t'].shape[::-1]
        cardClass=t['l']
        for pt in zip(*locations[::-1]):
            cv2.rectangle(test1gray, pt, (pt[0]+w, pt[1]+h),(0,255,255),2)
        #print(cardClass)
        break

#width, height = test1gray.shape[::-1]
#r = 1000.0 / test1gray.shape[1]
#dim = (1000, int(test1gray.shape[0] * r))
#test1gray = cv2.resize(test1gray,dim, interpolation = cv2.INTER_AREA)
#print(test1gray.shape)
#print(locations)
if cardClass=="h":
    print("Hearts")
elif cardClass=="s":
    print("Spades")
elif cardClass=="c":
    print("Clubs")
elif cardClass=="d":
    print("Diamonds")


cv2.imshow('detected',test1gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
