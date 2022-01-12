
from numpy import array, max  


def my_filter(a):

    return a > 200






    #sp = array( ["123", "111", "444", "555"] ) 
    #m = array( [1, 2, 3, 4] )
#   MAP 
    #m =  array( list( map(str_to_number, sp) ) ) 
 #  m[:] = str_to_number( sp[:] ) # it does not work 
    #N = len( sp )
    #for i in range(N) : 
    #    m[i] = str_to_number( sp[i] )

#   FILTER
f =  array(( filter(my_filter, m) ) ) 
print (f)
#   REDUCE 
    r =  max(m) 
    #print(" array of strings = ", sp)
    #print(" MAP: array of numbers = ", m)
    #print(" FILTER: filtered array = ", f)
    #print(" REDUCE: max value = ", r