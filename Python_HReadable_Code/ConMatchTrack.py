__author__ = "InteliTech"
__copyright__ = "Copyright 2015, InteliTech Solutions"

import cv2
import itertools
from numpy import array, zeros, vstack, hstack, math, nan, argsort, median, \
    argmax, isnan, append
import scipy.cluster
import scipy.spatial
import time

import numpy as np
import helper


class CMT(object):
    SCALE_DETECT = True
    ROTATION_ESTIMATE = True
    CORRESPOND_TRACKER = 'BruteForce-Hamming'
    FEATURE_DETECTOR = 'BRISK'
    FEATURE_DESCRIPTOR = 'BRISK'
    DESCRIPTOR_LENGTH = 512
    THRESH_RATIO = 8 / 10
    CONFIDENCE_THRESH = .75
    OUTLIER_THRESHOLD = 20

    def cmt_init(self, img_gray0, tl, br):

        # Initialise detector, descriptor, matcher
        # And Get initial corners in the complemete image
        self.detector = cv2.FeatureDetector_create(self.FEATURE_DETECTOR)
        cv_keypts = self.detector.detect(img_gray0)
        self.descriptor = cv2.DescriptorExtractor_create(self.FEATURE_DESCRIPTOR)
        self.matcher = cv2.DescriptorMatcher_create(self.CORRESPOND_TRACKER)
        # Extract keypts that are in the BBOX as the chosen keypts
        ind = helper.init_bbox(cv_keypts, tl, br)
        initial_keypts = list(itertools.compress(cv_keypts, ind))
        initial_keypts, self.selected_features = self.descriptor.compute(img_gray0, initial_keypts)
        chosen_keypts = helper.cv2npkeypts(initial_keypts)
        num_chosen_keypts = len(initial_keypts)
        if num_chosen_keypts == 0: raise Exception('No keypts detected in highlighted region')
        # Remember keypts that are not in the rectangle as backdrop keypts
        backdrop_cv_keypts = list(itertools.compress(cv_keypts, ~ind))
        backdrop_cv_keypts, background_features = self.descriptor.compute(img_gray0, backdrop_cv_keypts)
        _ = helper.cv2npkeypts(backdrop_cv_keypts)
        # Assign each keypoint a class starting from 1, backdrop is 0
        self.selected_classes = array(range(num_chosen_keypts)) + 1
        backdrop_classes = zeros(len(backdrop_cv_keypts))
        # collect all backdrop and BBOX extracted features into database
        self.features_database = vstack((background_features, self.selected_features))
        # Same for classes
        self.database_classes = hstack((backdrop_classes, self.selected_classes))
        # Get all distances between selected keypts in squareform
        distance = scipy.spatial.distance.pdist(chosen_keypts)
        self.squareform = scipy.spatial.distance.squareform(distance)
        # Get all angles between selected keypts
        theeta = np.empty((num_chosen_keypts, num_chosen_keypts))
        for k1, i1 in zip(chosen_keypts, range(num_chosen_keypts)):
            for k2, i2 in zip(chosen_keypts, range(num_chosen_keypts)):
                # Compute vectors k1 to k2
                v = k2 - k1
                # Compute angle of this vector with respect to x axis
                angle = math.atan2(v[1], v[0])
                # Store angle
                theeta[i1, i2] = angle
        self.theeta = theeta
        # Calculate center of chossen keypts
        center = np.mean(chosen_keypts, axis=0)
        # Remember the rectangle coordinates relative to the center
        self.center2tl = np.array(tl) - center
        self.center2tr = np.array([br[0], tl[1]]) - center
        self.center2br = np.array(br) - center
        self.center2bl = np.array([tl[0], br[1]]) - center
        # Calculate springs of each keypoint
        self.springs = chosen_keypts - center
        # Set start image for tracking
        self.img_prev = img_gray0
        # Make keypts 'active' keypts
        self.active_keypts = np.copy(chosen_keypts)
        # Attach class information to active keypts
        self.active_keypts = hstack((chosen_keypts, self.selected_classes[:, None]))
        # Remember number of initial keypts
        self.num_initial_keypts = len(initial_keypts)


    def frame_analysis(self, img_gray):

        tracked_keypts, _ = helper.tracker(self.img_prev, img_gray, self.active_keypts)

        (center, scale_predict, rotation_estimate, tracked_keypts) = self.predict(tracked_keypts)
        # Detect keypts, compute descriptors
        cv_keypts = self.detector.detect(img_gray)
        cv_keypts, features = self.descriptor.compute(img_gray, cv_keypts)
        # Create list of active keypts
        active_keypts = zeros((0, 3))
        # Get the two best matches for each feature
        matches_all = self.matcher.knnMatch(features, self.features_database, 2)
        # Get all matches for selected features
        if not any(isnan(center)): selected_matches_all = self.matcher.knnMatch(features, self.selected_features,
                                                                                len(self.selected_features))
        # For each keypoint and its descriptor
        if len(cv_keypts) > 0: transformed_springs = scale_predict * helper.rotation(self.springs, -rotation_estimate)
        for i in range(len(cv_keypts)):
            # Retrieve keypoint location
            location = np.array(cv_keypts[i].pt)
            # First: Match over whole image
            # Compute distances to all descriptors
            crossmatch = matches_all[i]
            distances = np.array([m.distance for m in crossmatch])
            # Convert distances to confidence_score, without weight
            collective = 1 - distances / self.DESCRIPTOR_LENGTH
            classes = self.database_classes
            # Get best and second best index
            best_index = crossmatch[0].trainIdx
            secondBestInd = crossmatch[1].trainIdx
            # Compute distance propotion according to David Lowe
            proportion = (1 - collective[0]) / (1 - collective[1])
            # Extract class of best match
            keypoint_class = classes[best_index]
            # If distance proportion is ok and absolute distance is ok and keypoint class is not background
            if proportion < self.THRESH_RATIO and collective[0] > self.CONFIDENCE_THRESH and keypoint_class != 0:
                # Add keypoint to active keypts
                new_kpt = append(location, keypoint_class)
                active_keypts = append(active_keypts, array([new_kpt]), axis=0)
            # In a second step, try to match difficult keypts
            # If structural constraints are applicable
            if not any(isnan(center)):
                # Compute distances to initial descriptors
                crossmatch= selected_matches_all[i]
                distances = np.array([m.distance for m in crossmatch])
                # Re-order the distances based on indexing
                idxs = np.argsort(np.array([m.trainIdx for m in crossmatch]))
                distances = distances[idxs]
                # Convert distances to confidence score
                confidence_score = 1 - distances / self.DESCRIPTOR_LENGTH
                # Compute the keypoint location relative to the object center
                locrel = location - center
                # Compute the distances to all springs
                displacements = helper.L2norm(transformed_springs - locrel)
                # For each spring, calculate weight
                Wt = displacements < self.OUTLIER_THRESHOLD  # Could be smooth function
                collective = Wt * confidence_score
                classes = self.selected_classes
                # Sort in descending order
                sorted_conf = argsort(collective)[::-1]  # reverse
                # Get best and second best index
                best_index = sorted_conf[0]
                best_index2 = sorted_conf[1]
                # Compute distance proportion according to Lowe
                proportion = (1 - collective[best_index]) / (1 - collective[best_index2])
                # Extract class of best match
                keypoint_class = classes[best_index]
                # If distance proportion is ok and absolute distance is ok and keypoint class is not background
                if proportion < self.THRESH_RATIO and collective[best_index] > self.CONFIDENCE_THRESH and keypoint_class != 0:
                    # Add keypoint to active keypts
                    new_kpt = append(location, keypoint_class)
                    # Check whether same class already exists
                    if active_keypts.size > 0:
                        same_class = np.nonzero(active_keypts[:, 2] == keypoint_class)
                        active_keypts = np.delete(active_keypts, same_class, axis=0)
                    active_keypts = append(active_keypts, array([new_kpt]), axis=0)
                    # If some keypts have been tracked
                    # Extract the keypoint classes
        if tracked_keypts.size > 0: tracked_classes = tracked_keypts[:, 2]
        # If there already are some active keypts
        if active_keypts.size > 0:
            # Add all tracked keypts that have not been matched
            associated_classes = active_keypts[:, 2]
            missing = ~np.in1d(tracked_classes, associated_classes)
            active_keypts = append(active_keypts, tracked_keypts[missing, :], axis=0)
        # Else use all tracked keypts
        else: active_keypts = tracked_keypts

    # Update object state estimate
        _ = active_keypts
        self.center = center
        self.scale_predict = scale_predict
        self.rotation_estimate = rotation_estimate
        self.tracked_keypts = tracked_keypts
        self.active_keypts = active_keypts
        self.img_prev = img_gray
        self.cv_keypts = cv_keypts
        _ = time.time()
        self.tl = (nan, nan)
        self.tr = (nan, nan)
        self.br = (nan, nan)
        self.bl = (nan, nan)
        self.bb = array([nan, nan, nan, nan])
        self.has_result = False
        if not any(isnan(self.center)) and self.active_keypts.shape[0] > self.num_initial_keypts / 10:
            self.has_result = True
            tl = helper.array2integer(center + scale_predict * helper.rotation(self.center2tl[None, :], rotation_estimate).squeeze())
            tr = helper.array2integer(center + scale_predict * helper.rotation(self.center2tr[None, :], rotation_estimate).squeeze())
            br = helper.array2integer(center + scale_predict * helper.rotation(self.center2br[None, :], rotation_estimate).squeeze())
            bl = helper.array2integer(center + scale_predict * helper.rotation(self.center2bl[None, :], rotation_estimate).squeeze())
        min_x = min((tl[0], tr[0], br[0], bl[0]))
        min_y = min((tl[1], tr[1], br[1], bl[1]))
        max_x = max((tl[0], tr[0], br[0], bl[0]))
        max_y = max((tl[1], tr[1], br[1], bl[1]))
        self.tl = tl
        self.tr = tr
        self.bl = bl
        self.br = br
        self.bb = np.array([min_x, min_y, max_x - min_x, max_y - min_y])

    def predict(self, keypts):
        center = array((nan, nan))
        scale_predict = nan
        med_rot = nan
        # At least 2 keypts are needed for scale
        if keypts.size > 1:
            # Extract the keypoint classes
            keypt_classes = keypts[:, 2].squeeze().astype(np.int)
            # Retain singular dimension
            if keypt_classes.size == 1:
                keypt_classes = keypt_classes[None]
            # Sort
            ind_sort = argsort(keypt_classes)
            keypts = keypts[ind_sort]
            keypt_classes = keypt_classes[ind_sort]
            # Get all combinations of keypts
            all_combs = array([val for val in itertools.product(range(keypts.shape[0]), repeat=2)])
            # But exclude comparison with itself
            all_combs = all_combs[all_combs[:, 0] != all_combs[:, 1], :]
            # Measure distance between allcombs[0] and allcombs[1]
            ind1 = all_combs[:, 0]
            ind2 = all_combs[:, 1]
            class_ind1 = keypt_classes[ind1] - 1
            class_ind2 = keypt_classes[ind2] - 1
            duplicate_classes = class_ind1 == class_ind2
            if not all(duplicate_classes):
                ind1 = ind1[~duplicate_classes]
                ind2 = ind2[~duplicate_classes]
                class_ind1 = class_ind1[~duplicate_classes]
                class_ind2 = class_ind2[~duplicate_classes]
                pts_allcombs0 = keypts[ind1, :2]
                pts_allcombs1 = keypts[ind2, :2]
                dists = helper.L2norm(pts_allcombs0 - pts_allcombs1)
                original_dists = self.squareform[class_ind1, class_ind2]
                scalechange = dists / original_dists
                # Compute angles
                theeta = np.empty((pts_allcombs0.shape[0]))
                v = pts_allcombs1 - pts_allcombs0
                theeta = np.arctan2(v[:, 1], v[:, 0])
                original_theeta = self.theeta[class_ind1, class_ind2]
                angle_diffs = theeta - original_theeta
                # Fix long way angles
                long_way_theeta = np.abs(angle_diffs) > math.pi
                angle_diffs[long_way_theeta] = angle_diffs[long_way_theeta] - np.sign(angle_diffs[long_way_theeta]) * 2 * math.pi
                scale_predict = median(scalechange)
                if not self.SCALE_DETECT:
                    scale_predict = 1;
                med_rot = median(angle_diffs)
                if not self.ROTATION_ESTIMATE:
                    med_rot = 0;
                keypoint_class = keypts[:, 2].astype(np.int)
                votes = keypts[:, :2] - scale_predict * (helper.rotation(self.springs[keypoint_class - 1], med_rot))
                # Remember all votes including outliers
                self.votes = votes
                # Compute pairwise distance between votes
                distance = scipy.spatial.distance.pdist(votes)
                # Compute linkage between pairwise distances
                linkage = scipy.cluster.hierarchy.linkage(distance)
                # Perform hierarchical distance-based clustering
                T = scipy.cluster.hierarchy.fcluster(linkage, self.OUTLIER_THRESHOLD, criterion='distance')
                # Count votes for each cluster
                cnt = np.bincount(T)  # Dummy 0 label remains
                # Get largest class
                Cmax = argmax(cnt)
                # Identify inliers (=members of largest class)
                inliers = T == Cmax
                # inliers = med_dists < OUTLIER_THRESHOLD
                # Remember outliers
                self.outliers = keypts[~inliers, :]
                # Stop tracking outliers
                keypts = keypts[inliers, :]
                # Remove outlier votes
                votes = votes[inliers, :]
                # Compute object center
                center = np.mean(votes, axis=0)
        return (center, scale_predict, med_rot, keypts)