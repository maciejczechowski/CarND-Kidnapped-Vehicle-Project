/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    weights = std::vector<double>(num_particles, 1.0);
    for (int i = 0; i < num_particles; ++i) {

        Particle particle{i, addNoise(x, std[0]), addNoise(y, std[1]), addNoise(theta, std[2])};
        particles.push_back(particle);

    }
    is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {

    for (auto &&particle: particles) {
        double xf, yf, thetaf;
        if (abs(yaw_rate) > std::numeric_limits<double>::epsilon()) {
            xf = particle.x + (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
            yf = particle.y + (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
            thetaf = particle.theta + yaw_rate * delta_t;
        } else {
            xf = particle.x + velocity * delta_t * cos(particle.theta);
            yf = particle.y + velocity * delta_t * sin(particle.theta);
            thetaf = particle.theta;
        }
        particle.x = addNoise(xf, std_pos[0]);
        particle.y = addNoise(yf, std_pos[1]);
        particle.theta = addNoise(thetaf, std_pos[2]);
    }
}


void ParticleFilter::dataAssociation(const vector<LandmarkObs> &predicted,
                                     vector<LandmarkObs> &observations) {

    // Brute force with O(n^2) complexity
    // possible area of improvement, as much better algorithms are known
    for (auto &&observation: observations) {
        double currentMinimumDistance = std::numeric_limits<double>::max();
        int currentId = -1;
        for (auto &&prediction: predicted) {

            double distance = dist(observation.x, observation.y, prediction.x, prediction.y);
            if (distance < currentMinimumDistance) {
                currentId = prediction.id;
                currentMinimumDistance = distance;

            }
        }
        observation.id = currentId;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {

    if (!has_map_of_landmarks) {
        for (auto &&mapPoint: map_landmarks.landmark_list) {
            map_of_landmarks[mapPoint.id_i] = mapPoint;
        }
        has_map_of_landmarks = true;
    }


    weights.clear();
    double meas_error = sqrt(std_landmark[0] * std_landmark[0] + std_landmark[1] + std_landmark[1]);
    // iterate over particle
    for (auto &&particle: particles) {

        //find out which map landmarks could be observed from sensor
        vector<LandmarkObs> predictions;
        for (auto &&mapPoint: map_landmarks.landmark_list) {
            double landmarkDistance = dist(particle.x, particle.y, mapPoint.x_f, mapPoint.y_f);
            if (landmarkDistance <= (sensor_range + meas_error)) {
                predictions.push_back({mapPoint.id_i, mapPoint.x_f, mapPoint.y_f});
            } else {
              //  std::cout << "Ignoring landmark since it's distance from us: " << particle.id << " is " << landmarkDistance << std::endl;

            }
        }

        //convert observations to map coordinate system
        vector<LandmarkObs> observationsOnMap = convertToMapCoordiates(observations, particle);

        //find out nearest neighbours of observations
        dataAssociation(predictions, observationsOnMap);

        double weight = 1.0;

        particle.associations = {};
        particle.sense_x = {};
        particle.sense_y = {};

        for (auto &&observation: observationsOnMap) {
            if (observation.id == -1) {
                continue;
            }
            particle.associations.push_back(observation.id);

            auto mapLandmark = map_of_landmarks[observation.id];
            particle.sense_x.push_back(observation.x);
            particle.sense_y.push_back(observation.y);

            double partial_weight = multiv_prob(std_landmark[0], std_landmark[1], observation.x, observation.y,
                                                mapLandmark.x_f, mapLandmark.y_f);
            weight *= partial_weight;

        }

        particle.weight = weight;
        weights.push_back(weight);
        SetAssociations(particle, particle.associations, particle.sense_x, particle.sense_y);
    }

}

vector<LandmarkObs> ParticleFilter::convertToMapCoordiates(const vector<LandmarkObs> &observations, const Particle &particle) {
    vector<LandmarkObs> observationsOnMap;
    for (auto &&observation: observations) {
        double xm = particle.x + (cos(particle.theta) * observation.x) - (sin(particle.theta) * observation.y);
        double ym = particle.y + (sin(particle.theta) * observation.x) + (cos(particle.theta) * observation.y);
        LandmarkObs mapLandmarkObs = {observation.id, xm, ym};
        observationsOnMap.push_back(mapLandmarkObs);
    }
    return observationsOnMap;
}

void ParticleFilter::resample() {

    std::vector<Particle> p3 = {};
    std::discrete_distribution<> distribution(weights.begin(), weights.end());

    for (int i = 0; i < num_particles; i++) {
        int index = distribution(gen);
        Particle chosen_one = particles[index];
        p3.push_back(chosen_one);
    }


    particles = p3;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y) {


    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
    vector<double> v;

    if (coord == "X") {
        v = best.sense_x;
    } else {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

double ParticleFilter::addNoise(double x, double std) {

    std::normal_distribution<double> dist_x(x, std);
    return dist_x(gen);
}