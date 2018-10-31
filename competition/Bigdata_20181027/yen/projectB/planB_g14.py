import planB
import TypeInfo

def main():
    typeInfo = TypeInfo.G14()
    avr, std = planB.autoMatch(typeInfo, 20, show=True, startIndex=10)
    print("avr: {}, std: {}".format(avr, std))

if __name__ == "__main__":
    main()
