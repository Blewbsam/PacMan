#pragma once

#include <SFML/Graphics.hpp>
#include "pacmanUI.hpp"
#include "globalsUI.hpp"
#include "../game/game.hpp"
#include "ghosts/ghostUI.hpp"



class Display {

private: 
    sf::RenderWindow* window;
    sf::VideoMode videoMode;
    sf::Event ev;
    GameState * gs;

    // game objects
    PacmanUI pacman;
    ChaserUI chaser;
    AmbusherUI ambusher;
    FickleUI fickle;
    StupidUI stupid;

    sf::RectangleShape empty;
    sf::CircleShape pellet;
    sf::CircleShape powerPellet;

    void initWindow();
    void initGameObjects();



    // check for and handle triggered events.
    void pollEvents();

    void updateGhosts();
    void updateGhost(GhostUI& ghost);

    // hanlding ghost collsions is done in UI for smoother experience
    void checkGhostCollisions();

    void moveGhosts();
    void moveGhost(GhostUI& ghost);

    void handleTeleports();
    void handleTeleport(AgentUI& agent);


    // following set of functions set agent in approriat position to be displayed
    void renderPacman();
    void renderGhosts();

    // use the grid_t structure to render display
    void renderMaze(); 

    void scaleSprite(sf::Sprite &sprite);

    void selectWall(sf::Sprite &sprite,grid_t& grid,size_t x, size_t y,size_t height, size_t width);

    void gameLost();

public:
    Display(GameState * gameState);
    ~Display();

    // listens and reponses to events
    // move pieces to determined directions
    void update();
    // render required portions onto the display
    void render();

    

    void step(int i);
    void setPacmanDir(Direction dir);

    bool pacmanContainedInCell();

    // returns wether the window is open or not
    bool running() const;

    sf::Image getScreenshot();
};

