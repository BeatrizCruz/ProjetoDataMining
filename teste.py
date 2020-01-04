##
# Create a figure space matrix consisting of 3 columns and 2 rows
#
# Here is a useful template to use for working with subplots.
#
##################################################################
variaveis=[variaveis]
clusters = nº clusters
fig, ax = plt.subplots(figsize=(10,5), ncols=2, nrows=len(variaveis))

left   =  0.125  # the left side of the subplots of the figure
right  =  0.9    # the right side of the subplots of the figure
bottom =  0.1    # the bottom of the subplots of the figure
top    =  0.9    # the top of the subplots of the figure
wspace =  .5     # the amount of width reserved for blank space between subplots
hspace =  1.1    # the amount of height reserved for white space between subplots

# This function actually adjusts the sub plots using the above paramters
plt.subplots_adjust(
    left    =  left,
    bottom  =  bottom,
    right   =  right,
    top     =  top,
    wspace  =  wspace,
    hspace  =  hspace
)

# The amount of space above titles
y_title_margin = 1.2

plt.suptitle("Comparing Clusters", y = 1.09, fontsize=20)
i=0
for var in variaveis:

    sns.catplot(var,col="labelsKmeansEngage", data=dfEngageKmeans, kind="count", ax=ax[i][0])
    sns.distplot(df[var], col="labelsKmeansEngage",kde = False, ax = ax[i][1])
    i+=1

#sns.distplot(df['stand_bathrooms'],  kde = False, ax=ax[0][2])

# Set all labels on the row axis of subplots for bathroom data to "bathrooms"
