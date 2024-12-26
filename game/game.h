

#ifndef GAME
#define GAME    

#include "globals.h"
#include "ghost.h"
#include "pacman.h"
#include "maze.h"
#include "ghostAI.h"
#include "globals.h"


#include <vector>



// Pointer to created Ghosts fond on maze
struct Ghosts {
    FickleGhost * fickle_p;
    ChaserGhost * chaser_p;
    AmbusherGhost * ambusher_p;
    StupidGhost * stupid_p;
};



class GameState {
    Maze * maze_p;
    Pacman * pacman_p;
    Ghosts ghosts;
    GhostAI ghostAI;
    unsigned int score;
    bool gameOver;

public:
    GameState();
    ~GameState();

    // getters used to get position of agents:
    Position getPacmanPos() const;
    Direction getPacmanDir() const;
    Position getFicklePos() const;
    Direction getFickleDir() const;
    Position getAmbusherPos() const;
    Direction getAmbusherDir() const;
    Position getStupidPos() const;
    Direction getStupidDir() const;
    Position getChaserPos() const;
    Direction getChaserDir() const;

    // maze_p accessors for communicating with GhostAI
    grid_t getGrid() const;
    int getGridWidth() const;
    int getGridHeight() const;





    // changes position of pacman to given pos.
    void updatePacmanPos(Position pos);

    // assesses wether pacman moving in given direction encounters a wall.
    bool validPacmanMove(Direction dir) const;

    // Changes direction of pacman to given direction
    void changePacmanDir(Direction dir);

    // set the directon of each ghost given the board and pacman positions.
    void generateGhostMoves();

    // set position of given ghost
    void updateGhostPos(Position pos,GhostType type);

    // set state of type to given state
    void updateGhostState(GhostType type, GhostState state);

    // handle all plausible collisions.
    void handleCollisions();

    // returns true if jump possible
    bool jumpAvail(Position pos);
    // takes current pos and returns what it would become
    // requires jumpAvail to return true
    Position jumpPortal(Position pos);

    // returns valid positions that the ghost can move in.
    std::vector<Position> getValidPositions(Position ghostPos, Direction ghostDir) const;
    // Return neighbors of position on grid which are not walls.
    std::vector<Position> getValidNeighbours(Position pos) const;





private:

    // Iterators over all cells to find position of specifiedagentCell being one of: 
    // {GHOST_FICKLE,GHOST_CHASER, GHOST_AMBUSER, GHOST_STUPID}
    // To be used in constructor only
    Position getAgentPositionBrute(Cell agentCell ,const std::vector<std::vector<Cell >> grid);    

    // check if there is a pellet at pacman's position
    // and remove it if there is so
    void handlePelletCollision();

    // return true if items are at same index
    bool checkAgentCollison(Agent a1, Agent a2);

};




#endif 
