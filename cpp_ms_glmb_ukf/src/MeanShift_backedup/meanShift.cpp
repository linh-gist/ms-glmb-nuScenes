#include <vector>
#include <iostream>
#include <cmath>

#include "Point.cpp"
#include "Cluster.cpp"
#include "ClustersBuilder.cpp"
//#include "meanShift.cpp"
#define MAX_ITERATIONS 100


std::vector<std::vector<float>> meanShift(const std::vector<std::vector<float>> &arrays, float bandwidth, std::vector<int> sensors) {
    std::vector<Point> points;
    for(int i=0; i<arrays.size();i++){
points.emplace_back(Point(arrays[i]));
    }
    ClustersBuilder builder = ClustersBuilder(points, 0.4);
    long iterations = 0;
    unsigned long dimensions = points[0].dimensions();
    float radius = bandwidth * 3;
    float doubledSquaredBandwidth = 2 * bandwidth * bandwidth;
//    std::vector<int> sensors = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
//                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
//                                2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6};
    int num_sensors = 7;
    while (!builder.allPointsHaveStoppedShifting() && iterations < MAX_ITERATIONS) {

#pragma omp parallel for default(none) \
shared(points, dimensions, builder, bandwidth, radius, doubledSquaredBandwidth, sensors, num_sensors) \
schedule(dynamic)

        for (long i = 0; i < points.size(); ++i) {
            if (builder.hasStoppedShifting(i))
                continue;

            Point newPosition(dimensions);
            std::vector<bool> checkSensors = std::vector<bool>(num_sensors, true);
            std::vector<Point> checkPoints = std::vector<Point>(num_sensors);
            std::vector<float> checkDis = std::vector<float>(num_sensors, std::numeric_limits<float>::max());
            Point pointToShift = builder.getShiftedPoint(i);
            float totalWeight = 0.0;
//            for (auto &point : points) {
            for (long j = 0; j < points.size(); ++j) {
                Point point = points[j];
                float distance = pointToShift.euclideanDistance(point);
                if (distance <= radius && checkDis[sensors[j]] > distance){
                    checkDis[sensors[j]] = distance;
                    checkPoints[sensors[j]] = point;
                }

//                if (distance <= radius && checkSensors[sensors[j]]) {
//                    float gaussian = std::exp(-(distance * distance) / doubledSquaredBandwidth);
//                    newPosition += point * gaussian;
//                    totalWeight += gaussian;
//                    checkSensors[sensors[j]] = false;
//                }
            }
            for(int k=0; k<checkDis.size(); k++){
                if (checkDis[k] < std::numeric_limits<float>::max()) {
                    float distance = checkDis[k];
                    Point point = checkPoints[k];
                    float gaussian = std::exp(-(distance * distance) / doubledSquaredBandwidth);
                    newPosition += point * gaussian;
                    totalWeight += gaussian;
                }
            }
//            for(bool se:checkSensors){
//                std::cout<<se<<" ";
//            }std::cout<<std::endl;

            // the new position of the point is the weighted average of its neighbors
            newPosition /= totalWeight;
            builder.shiftPoint(i, newPosition);
        }
        ++iterations;
    }
    if (iterations == MAX_ITERATIONS)
        std::cout << "WARNING: reached the maximum number of iterations" << std::endl;
    std::vector<Cluster> clusters = builder.buildClusters(sensors);
    std::vector<std::vector<float>> outputs;
    for(int i=0; i<clusters.size(); i++){
    outputs.emplace_back(clusters[i].getCentroid().values);
    std::cout<<i<<" "<<clusters[i].getCentroid().values[0]<<" "<<clusters[i].getCentroid().values[1]<<endl;

    }
    for (int i = 0; i < clusters.size(); i++) {
        std::cout << i << " ";
        for (int j = 0; j < num_sensors; j++)
            if (clusters[i].checkDis[j] < std::numeric_limits<float>::max()) {
                std::cout << j + 1 << " ";
            } else { std::cout << 0 << " "; }
        std::cout << std::endl;
    }
    return outputs;
}

