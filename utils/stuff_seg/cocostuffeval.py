__author__ = 'hcaesar'

import numpy as np
import datetime
import time
from .cocostuffhelper import cocoSegmentationToSegmentationMap

class COCOStuffeval:
    # Internal functions for evaluating stuff segmentations against a ground-truth.
    #
    # The usage for COCOStuffeval is as follows:
    #  cocoGt=..., cocoRes=...            # load dataset and results
    #  E = COCOStuffeval(cocoGt, cocoRes) # initialize COCOStuffeval object
    #  E.params.imgIds = ...              # set parameters as desired
    #  E.evaluate()                       # run per image evaluation
    #  E.summarize()                      # display summary metrics of results
    # For example usage see pycocostuffEvalDemo.py.
    #
    # Note: Our evaluation has to take place on all classes. If we remove one class
    # from evaluation, it is not clear what should happen to pixels for which a
    # method output that class (but where the ground-truth has a different class).
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #
    # evaluate(): evaluates segmentations on each image and
    # stores the results in the dictionary 'eval' with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  confusion  - confusion matrix used for the final metrics
    #
    # summarize(): computes and prints the evaluation metrics.
    # results are printed to stdout and stored in:
    #  stats      - a numpy array of the evaluation metrics (mean IOU etc.)
    #  statsClass - a dict that stores per-class results in ious and maccs
    #
    # See also coco, mask, pycocostuffDemo, pycocostuffEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]

    def __init__(self, cocoGt, cocoRes, stuffStartId=92, stuffEndId=182, addOther=True):
        '''
        Initialize COCOStuffeval using COCO APIs for gt and dt
        :param cocoGt: COCO object with ground truth annotations
        :param cocoRes: COCO object with detection results
        :param stuffStartId: id of the first stuff class
        :param stuffEndId: id of the last stuff class
        :param addOther: whether to use a other class
        :return: None
        '''
        self.cocoGt = cocoGt                # Ground truth COCO API
        self.cocoRes = cocoRes              # Result COCO API
        self.stuffStartId = stuffStartId    # Id of the first stuff class
        self.stuffEndId = stuffEndId        # Id of the last stuff class
        self.addOther = addOther            # Whether to add a class that subsumes all thing classes

        self.eval = {}                      # Accumulated evaluation results
        self.confusion = []                 # Confusion matrix that all metrics are computed on
        self.stats = []                     # Result summarization
        self.statsClass = {}                # Per-class results
        self.params = Params()              # Evaluation parameters
        self.params.imgIds = sorted(cocoGt.getImgIds()) # By default we use all images from the GT file
        self.catIds = range(stuffStartId, stuffEndId+addOther+1) # Take into account all stuff
                                                                 # classes and one 'other' class

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results in self.confusion.
        Note that this can take up to several hours.
        :return: None
        '''

        # Reset eval and print message
        tic = time.time()
        imgIds = self.params.imgIds
        print('Evaluating stuff segmentation on %d images and %d classes...' \
            % (len(imgIds), len(self.catIds)))

        # Check that all images in params occur in GT and results
        gtImgIds = sorted(set(self.cocoGt.getImgIds()))
        resImgIds = sorted(set([a['image_id'] for a in self.cocoRes.anns.values()]))
        missingInGt = [p for p in imgIds if p not in gtImgIds]
        missingInRes = [p for p in imgIds if p not in resImgIds]
        if len(missingInGt) > 0:
            raise Exception('Error: Some images specified in imgIds do not occur in the GT: %s' % missingInGt)
        if len(missingInRes) > 0:
            raise Exception('Error: Some evaluation images not found in the result: %s' % missingInRes)

        # Create confusion matrix
        labelCount = max([c for c in self.cocoGt.cats])
        confusion = np.zeros((labelCount, labelCount))
        for i, imgId in enumerate(imgIds):
            # if i+1 == 1 or i+1 == len(imgIds) or (i+1) % 10 == 0:
            #     print('Evaluating image %d of %d: %d' % (i+1, len(imgIds), imgId))
            confusion = self._accumulateConfusion(self.cocoGt, self.cocoRes, confusion, imgId)
        self.confusion = confusion

        # Set eval struct to be used later
        self.eval = {
            'params': self.params,
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'confusion': self.confusion
        }

        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def _accumulateConfusion(self, cocoGt, cocoRes, confusion, imgId):
        '''
        Accumulate the pixels of the current image in the specified confusion matrix.
        Note: For simplicity we do not map the labels to range [0, L-1],
              but keep the original indices when indexing 'confusion'.
        :param cocoGt: COCO object with ground truth annotations
        :param cocoRes: COCO object with detection results
        :param confusion: confusion matrix that will be modified
        :param imgId: id of the current image
        :return: confusion (modified confusion matrix)
        '''

        # Combine all annotations of this image in labelMapGt and labelMapRes
        labelMapGt  = cocoSegmentationToSegmentationMap(cocoGt,  imgId, includeCrowd=False)
        labelMapRes = cocoSegmentationToSegmentationMap(cocoRes, imgId, includeCrowd=False)

        # Check that the result has only valid labels
        # invalidLabels = [l for l in np.unique(labelMapRes) if l not in self.catIds]
        # if len(invalidLabels) > 0:
        #     raise Exception('Error: Invalid classes of image %s predicted in the result file: %s. Please insert only labels in the range [%d, %d]!'
        #     % (str(imgId), str(invalidLabels), min(self.catIds), max(self.catIds)))

        # Filter labels that are not in catIds (includes the 0 label)
        valid = np.reshape(np.in1d(labelMapGt, self.catIds), labelMapGt.shape)
        validGt = labelMapGt[valid].astype(int)
        validRes = labelMapRes[valid].astype(int)

        # Gather annotations in confusion matrix
        #for g, d in zip(validGt, validRes):
        #    confusion[g-1, d-1] += 1

        # Much faster version using np.unique
        n = confusion.shape[0] + 1  # Arbitrary number > labelCount
        map_for_count = validGt * n + validRes
        vals, cnts = np.unique(map_for_count, return_counts=True)
        for v, c in zip(vals, cnts):
            g = v // n
            d = v % n
            confusion[g - 1, d - 1] += c

        return confusion

    def summarize(self):
        '''
        Compute and display the metrics for leaf nodes and super categories.
        :return: tuple of (general) stats and (per-class) statsClass
        '''

        # Check if evaluate was run and then compute performance metrics
        if not self.eval:
            raise Exception('Error: Please run evaluate() first!')

        # Compute confusion matrix for supercategories
        confusion = self.confusion
        confusionSup = self._getSupCatConfusion(confusion)

        # Compute performance
        [miou, fwiou, macc, pacc, ious, maccs] = self._computeMetrics(confusion)
        [miouSup, fwiouSup, maccSup, paccSup, iousSup, maccsSup] = self._computeMetrics(confusionSup)

        # Store metrics
        stats = np.zeros((8,))
        stats[0] = self._printSummary('Mean IOU', 'leaves', miou)
        stats[1] = self._printSummary('FW IOU', 'leaves', fwiou)
        stats[2] = self._printSummary('Mean accuracy', 'leaves', macc)
        stats[3] = self._printSummary('Pixel accuracy', 'leaves', pacc)
        stats[4] = self._printSummary('Mean IOU', 'supercats', miouSup)
        stats[5] = self._printSummary('FW IOU', 'supercats', fwiouSup)
        stats[6] = self._printSummary('Mean accuracy', 'supercats', maccSup)
        stats[7] = self._printSummary('Pixel accuracy', 'supercats', paccSup)

        # Store statsClass
        statsClass = {
            'ious': ious,
            'maccs': maccs,
            'iousSup': iousSup,
            'maccsSup': maccsSup
        }
        self.stats, self.statsClass = stats, statsClass

        return stats, statsClass

    def _getSupCatConfusion(self, confusion):
        '''
        Maps the leaf category confusion matrix to a super category confusion matrix.
        :param confusion: leaf category confusion matrix
        :return: confusionSup (super category confusion matrix)
        '''

        # Retrieve supercategory mapping
        supCats = [c['supercategory'] for c in self.cocoGt.cats.values()]
        supCatsUn = sorted(set(supCats))
        keys = supCatsUn
        vals = range(0, len(supCatsUn))
        supCatMap = dict(zip(keys, vals))
        supCatIds = [supCatMap[s] for s in supCats]
        supCatCount = len(supCatsUn)

        # Compute confusion matrix for supercategories
        confusionSup = np.zeros((supCatCount, supCatCount))
        for supCatIdA in range(0, supCatCount):
            for supCatIdB in range(0, supCatCount):
                curLeavesA = np.where([s == supCatIdA for s in supCatIds])[0] + self.stuffStartId - 1
                curLeavesB = np.where([s == supCatIdB for s in supCatIds])[0] + self.stuffStartId - 1
                confusionLeaves = confusion[curLeavesA, :]
                confusionLeaves = confusionLeaves[:, curLeavesB]
                confusionSup[supCatIdA, supCatIdB] = confusionLeaves.sum()
        assert confusionSup.sum() == confusion.sum()

        return confusionSup

    def _computeMetrics(self, confusion):
        '''
        Compute evaluation metrics given a confusion matrix.
        :param confusion: any confusion matrix
        :return: tuple (miou, fwiou, macc, pacc, ious, maccs)
        '''

        # Init
        labelCount = confusion.shape[0]
        ious = np.zeros((labelCount))
        maccs = np.zeros((labelCount))
        ious[:] = np.NAN
        maccs[:] = np.NAN

        # Get true positives, positive predictions and positive ground-truth
        total = confusion.sum()
        if total <= 0:
            raise Exception('Error: Confusion matrix is empty!')
        tp = np.diagonal(confusion)
        posPred = confusion.sum(axis=0)
        posGt = confusion.sum(axis=1)

        # Check which classes have elements
        valid = posGt > 0
        iousValid = np.logical_and(valid, posGt + posPred - tp > 0)

        # Compute per-class results and frequencies
        ious[iousValid] = np.divide(tp[iousValid], posGt[iousValid] + posPred[iousValid] - tp[iousValid])
        maccs[valid] = np.divide(tp[valid], posGt[valid])
        freqs = np.divide(posGt, total)

        # Compute evaluation metrics
        miou = np.mean(ious[iousValid])
        fwiou = np.sum(np.multiply(ious[iousValid], freqs[iousValid]))
        macc = np.mean(maccs[valid])
        pacc = tp.sum() / total

        return miou, fwiou, macc, pacc, ious, maccs

    def _printSummary(self, titleStr, classStr, val):
        '''
        Prints the current metric title, class type and value.
        :param titleStr: a string that represents the name of the metric
        :param classStr: the type of classes the metric was performed on (leaves/supercategories)
        :param val: the value of the metric
        '''
        iStr = ' {:<14} @[ classes={:>8s} ] = {:0.4f}'
        print(iStr.format(titleStr, classStr, val))

class Params:
    '''
    Params for coco stuff evaluation api
    '''

    def __init__(self):
        self.imgIds = []
