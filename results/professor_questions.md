# Questions a professor is likely to ask about this project

Questions only, no answers. Work through them out loud before the meeting.

## Data and segmentation
1. How did you find the individual shapes in a photo? Why not threshold on a fixed brightness?
2. Why take the minimum of the R, G, B channels instead of converting to grayscale?
3. What is a morphological closing, and why does subtracting it from the image cancel lighting?
4. What kinds of photos broke the segmentation, and what did you do about them?
5. How many distinct drawings do you really have, given that some photos are re-shots?

## Split and evaluation
6. Why did you split by photo rather than by individual crop? What would go wrong otherwise?
7. Why is photo 11 a good test set? What would make it a bad one?
8. Your test set has 49 images. What is the smallest accuracy difference you could detect with it?
9. All three classifiers scored 100% on the test photo. Does that mean they are equally good?
10. What accuracy would "always guess X" get on each split, and why does that matter?

## Preprocessing and augmentation
11. Why 32x32? What goes wrong at 64x64 for the MLP specifically?
12. Why does the model see a grayscale ink image rather than the colour crop?
13. Why did you not binarize by default? What did the preview show?
14. For each augmentation, what real-world variation does it protect against?
15. Why is colour jitter applied before the ink conversion rather than after?
16. Why are horizontal flips safe here but not for, say, digit recognition?

## Classifier 1, the manual perceptron
17. What is a perceptron computing, in one equation? What does the bias do?
18. Why do the features divide by the average ink density of the crop?
19. Why was the corners feature useless? How did you find out?
20. Which shapes does the perceptron get wrong, and why can no choice of weights fix them?
21. What would you have to add to fix them: a weight, or something else?

## Classifier 2, the MLP
22. Why can't two linear layers in a row do more than one? What does ReLU change?
23. The MLP has 139,521 weights and 510 training images. Why does it not memorise them?
24. Why is training accuracy below validation accuracy in your curves?
25. Why is the training loss curve jagged and the validation curve smooth?
26. What did the no-augmentation run look like, and what is that pattern called?
27. Why binary cross-entropy rather than accuracy as the loss? Why one output rather than two?
28. What does Adam do differently from plain gradient descent?
29. What does weight decay do on each step, and why does that help generalisation?
30. Where is dropout applied, and why is it switched off at evaluation time?
31. What is early stopping, and why does it depend on the split being by photo?

## Classifier 3, the CNN
32. What does a convolution compute? What two things does it have that a dense layer lacks?
33. What does max pooling do, and why does it help with shapes drawn in slightly different places?
34. Trace the tensor shape through your CNN from 1x32x32 to the single output.
35. Of the CNN's 136,001 weights, only 4,800 are in the convolutions. Where are the rest, and how would you cut them by four?
36. The CNN reached 96% validation accuracy in 10 epochs; the MLP took 43. Why?
37. Why did you put dropout only before the dense layer and not inside the conv blocks?
38. Both models tie on validation. On what evidence would you still prefer the CNN?

## Honest limitations
39. What would break these models in class that your data did not cover?
40. If the blindfolded drawer produces a very flat oval, which classifiers fail and why?
41. If you had a week more, what would you change first?
