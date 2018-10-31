import planB
import TypeInfo

def main():
    typeInfo = TypeInfo.G57()
    avr, std = planB.autoMatch(typeInfo, 15, show=True, startIndex=30)
    print("avr: {}, std: {}".format(avr, std))

if __name__ == "__main__":
    main()
