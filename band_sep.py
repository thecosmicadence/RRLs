import math

# Routine for separating G band, RP band and BP band magnitudes

def bandsep(x,band,spec1,spec2):
        m=0
        for k in range(len(spec1)):
            if(spec1[k][2]==band and m<=x):  
                spec2[m][0]=spec1[k][3]
                spec2[m][1]=spec1[k][4]
                if(math.isnan(spec2[m][0])==True and math.isnan(spec2[m][1])==True):
                    spec2[m][0]=0
                    spec2[m][1]=0
                m+=1