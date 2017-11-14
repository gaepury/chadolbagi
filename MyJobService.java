package com.example.g.cardet;

import android.app.job.JobParameters;
import android.app.job.JobService;

/**
 * Created by nohjunho on 2017-11-06.
 */

public class MyJobService extends JobService {
    @Override
    public boolean onStartJob(JobParameters params) {
        return false;
    }

    @Override
    public boolean onStopJob(JobParameters params) {
        return false;
    }
}
