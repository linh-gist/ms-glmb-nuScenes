#ifndef MEANSHIFT_CLUSTER_HPP
#define MEANSHIFT_CLUSTER_HPP

#include <vector>

#include "Point.hpp"


class Cluster {
public:
    explicit Cluster(Point centroid);

    Point getCentroid() const;

    void addPoint(Point point, int sensor, float distance);

    long getSize() const;

    std::vector<Point>::iterator begin();

    std::vector<Point>::iterator end();

    float getSse() const;

public:
    std::vector<Point> points;
    std::vector<Point> checkPoints;
    std::vector<float> checkDis;
    Point centroid;
};


#endif //MEANSHIFT_CLUSTER_HPP
