class Solution:
    def intToRoman(self, num: int) -> str:
        dic = {1:'I',
               4:'IV',
               5:'V',
               9:'IX',
               10:'X',
               40:'XL',
               50:'L',
               90:'XC',
               100:'C',
               400:'CD',
               500:'D',
               900:'CM',
               1000:'M'}
        out = ''
        for x in list(dic.keys())[::-1]:
            if x <= num:
                cnt = num //x
                num = num%x
                print(cnt,num,x)
                for k in range(cnt):
                    out+=dic[x]
                    print(out)
                    
s = Solution()

s.intToRoman(5)