package com.example.g.cardet;

import android.content.Context;
import android.util.Log;

import com.example.g.cardet.clusterers.ClusterEvaluation;
import com.example.g.cardet.clusterers.DensityBasedClusterer;
import com.example.g.cardet.clusterers.EM;
import com.example.g.cardet.core.Instances;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.util.Arrays;

import static android.R.attr.data;

public class ClusteringDemo implements Serializable {
    EM clusterer;

    public ClusteringDemo(Context ctx, InputStream is) throws Exception {
        Instances data;
        String[] options;
        long start = System.currentTimeMillis();
        Log.i("clustering", System.currentTimeMillis() + ": Clusterer Build Start");

        data = new Instances(new BufferedReader(new InputStreamReader(is)));

        options = new String[2];
        options[0] = "-l";
        options[1] = "100";
        clusterer = new EM();
        clusterer.setOptions(options);
        options = new String[2];
        options[0] = "-N";
        options[1] = "7";
        clusterer.setOptions(options);
        clusterer.buildClusterer(data);
        double[] d=clusterer.clusterPriors();
        Log.i("clu", Arrays.toString(d));

        Log.i("clustering", System.currentTimeMillis() + ": Clusterer Builded.");
        long end = System.currentTimeMillis();
        Log.i("클러스터링 빌드 시간",((end-start)/1000.0) + "s");
    }

    public double[] test(Context ctx, InputStream is) throws Exception {
        ClusterEvaluation eval;
        Instances data;
        data = new Instances(new BufferedReader(new InputStreamReader(is)));

//        Log.i("clustering", System.currentTimeMillis() + ": Evaluating Start");

        eval = new ClusterEvaluation();
        eval.setClusterer(clusterer);
        double[] result = eval.evaluateClusterer(data);
//        Log.i("clustering", "# of clusters: " + eval.getNumClusters());
//        Log.i("clustering", eval.clusterResultsToString());
//        Log.i("clustering", System.currentTimeMillis() + ": Clusterer Evaluated.");

        return result;
    }
}
