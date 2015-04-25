# Write data from various R example data sets to text
library(MASS)

# Measurements of the annual flow of the river Nile at Ashwan 1871–1970.
write(Nile,"Nile.dat",ncolumns=1)

# Data on patients diagnosed with AIDS in Australia before 1 July 1991. Age (years) at diagnosis.
write(Aids2$age,"Aids2$age.dat",ncolumns=1)

# Housind data from suburban boston
# crim: per capita crime rate by town.
# medv: median value of owner-occupied homes in \$1000s.
# rm: average number of rooms per dwelling.
write(log10(Boston$crim), "Boston$crim_log10.dat",ncolumns=1)
write(log10(Boston$medv), "Boston$medv_log10.dat",ncolumns=1)
write(Boston$rm, "Boston$rm.dat",ncolumns=1)

# The Melanoma data frame has data on 205 patients in Denmark with malignant melanoma.
# thickness: tumour thickness in mm.
# age: age in years.
write(log10(Melanoma$thickness), "Melanoma$thickness_log10.dat",ncolumns=1)
write(log10(Melanoma$age), "Melanoma$age.dat",ncolumns=1)

# Diabetes in Pima Indian Women
write(c(Pima.tr$glu,Pima.te$glu), "Pima$glu.dat",ncolumns=1)
write(c(Pima.tr$bp,Pima.te$bp), "Pima$bp.dat",ncolumns=1)
write(c(Pima.tr$skin,Pima.te$skin), "Pima$skin.dat",ncolumns=1)
write(c(Pima.tr$bmi,Pima.te$bmi), "Pima$bmi.dat",ncolumns=1)
write(c(Pima.tr$age,Pima.te$age), "Pima$age.dat",ncolumns=1)

