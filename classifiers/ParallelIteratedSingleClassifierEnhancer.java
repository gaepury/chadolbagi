/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    ParallelIteratedSingleClassifierEnhancer.java
 *    Copyright (C) 2009 University of Waikato, Hamilton, New Zealand
 *
 */

package com.example.g.cardet.classifiers;

import android.util.Log;

import java.util.Enumeration;
import java.util.Vector;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import com.example.g.cardet.core.Instances;
import com.example.g.cardet.core.Option;
import com.example.g.cardet.core.Utils;

/**
 * Abstract utility class for handling settings common to
 * meta classifiers that build an ensemble in parallel from a single
 * base learner.
 *
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 * @version $Revision: 6264 $
 */
public abstract class ParallelIteratedSingleClassifierEnhancer extends
    IteratedSingleClassifierEnhancer {

  /** For serialization */
  private static final long serialVersionUID = -5026378741833046436L;

  /** The number of threads to have executing at any one time */
  protected int m_numExecutionSlots = 1;

  /** Pool of threads to train models with */
  protected transient ThreadPoolExecutor m_executorPool;

  /** The number of classifiers completed so far */
  protected int m_completed;

  /**
   * The number of classifiers that experienced a failure of some sort
   * during construction
   */
  protected int m_failed;

  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options.
   */
  public Enumeration listOptions() {

    Vector newVector = new Vector(2);

    newVector.addElement(new Option(
              "\tNumber of execution slots.\n"
              + "\t(default 1 - i.e. no parallelism)",
              "num-slots", 1, "-num-slots <num>"));

    Enumeration enu = super.listOptions();
    while (enu.hasMoreElements()) {
      newVector.addElement(enu.nextElement());
    }
    return newVector.elements();
  }

  /**
   * Parses a given list of options. Valid options are:<p>
   *
   * -Z num <br>
   * Set the number of execution slots to use (default 1 - i.e. no parallelism). <p>
   *
   * Options after -- are passed to the designated classifier.<p>
   *
   * @param options the list of options as an array of strings
   * @exception Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception {

    String iterations = Utils.getOption("num-slots", options);
    if (iterations.length() != 0) {
      setNumExecutionSlots(Integer.parseInt(iterations));
    } else {
      setNumExecutionSlots(1);
    }

    super.setOptions(options);
  }

  /**
   * Gets the current settings of the classifier.
   *
   * @return an array of strings suitable for passing to setOptions
   */
  public String [] getOptions() {

    String [] superOptions = super.getOptions();
    String [] options = new String [superOptions.length + 2];

    int current = 0;
    options[current++] = "-num-slots";
    options[current++] = "" + getNumExecutionSlots();

    System.arraycopy(superOptions, 0, options, current,
                     superOptions.length);

    return options;
  }

  /**
   * Set the number of execution slots (threads) to use for building the
   * members of the ensemble.
   *
   * @param numSlots the number of slots to use.
   */
  public void setNumExecutionSlots(int numSlots) {
    m_numExecutionSlots = numSlots;
  }

  /**
   * Get the number of execution slots (threads) to use for building
   * the members of the ensemble.
   *
   * @return the number of slots to use
   */
  public int getNumExecutionSlots() {
    return m_numExecutionSlots;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numExecutionSlotsTipText() {
    return "The number of execution slots (threads) to use for " +
      "constructing the ensemble.";
  }

  /**
   * Stump method for building the classifiers
   *
   * @param data the training data to be used for generating the ensemble
   * @exception Exception if the classifier could not be built successfully
   */
  public void buildClassifier(Instances data) throws Exception {
    super.buildClassifier(data);

    if (m_numExecutionSlots < 1) {
      throw new Exception("Number of execution slots needs to be >= 1!");
    }

    if (m_numExecutionSlots > 1) {
      startExecutorPool();
    }
    m_completed = 0;
    m_failed = 0;
  }

  /**
   * Start the pool of execution threads
   */
  protected void startExecutorPool() {
    if (m_executorPool != null) {
      m_executorPool.shutdownNow();
    }

    m_executorPool = new ThreadPoolExecutor(m_numExecutionSlots, m_numExecutionSlots,
        120, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());
  }

  private synchronized void block(boolean tf) {
    if (tf) {
      try {
        wait();
      } catch (InterruptedException ex) {
      }
    } else {
      notifyAll();
    }
  }

  /**
   * Does the actual construction of the ensemble
   *
   * @throws Exception if something goes wrong during the training
   * process
   */
  protected synchronized void buildClassifiers() throws Exception {

    for (int i = 0; i < m_Classifiers.length; i++) {
      if (m_numExecutionSlots > 1) {
        final Classifier currentClassifier = m_Classifiers[i];
        final int iteration = i;
        if (m_Debug) {
          Log.i("clustering", "Training classifier (" + (i +1) + ")");
        }
        Runnable newTask = new Runnable() {
          public void run() {
            try {
              currentClassifier.buildClassifier(getTrainingSet(iteration));
              completedClassifier(iteration, true);
            } catch (Exception ex) {
              ex.printStackTrace();
              completedClassifier(iteration, false);
            }
          }
        };

        // launch this task
        m_executorPool.execute(newTask);
      } else {
        m_Classifiers[i].buildClassifier(getTrainingSet(i));
      }
    }

    if (m_numExecutionSlots > 1 && m_completed + m_failed < m_Classifiers.length) {
      block(true);
    }
  }

  /**
   * Records the completion of the training of a single classifier. Unblocks if
   * all classifiers have been trained.
   *
   * @param iteration the iteration that has completed
   * @param success whether the classifier trained successfully
   */
  protected synchronized void completedClassifier(int iteration,
      boolean success) {

    if (!success) {
      m_failed++;
      if (m_Debug) {
        System.err.println("Iteration " + iteration + " failed!");
      }
    } else {
      m_completed++;
    }

    if (m_completed + m_failed == m_Classifiers.length) {
      if (m_failed > 0) {
        if (m_Debug) {
          System.err.println("Problem building classifiers - some iterations failed.");
        }
      }

      // have to shut the pool down or program executes as a server
      // and when running from the command line does not return to the
      // prompt
      m_executorPool.shutdown();
      block(false);
    }
  }

  /**
   * Gets a training set for a particular iteration. Implementations need
   * to be careful with thread safety and should probably be synchronized
   * to be on the safe side.
   *
   * @param iteration the number of the iteration for the requested training set
   * @return the training set for the supplied iteration number
   * @throws Exception if something goes wrong.
   */
  protected abstract Instances getTrainingSet(int iteration) throws Exception;
}
