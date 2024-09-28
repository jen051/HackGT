import React from "react";
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";
import FightingStance from "./FightingStance";
import Punch from "./Punch"

function App() {
    return (
        <Router>
            <div>
                <h1>Self Defense Training</h1>
                <Routes>
                    <Route
                        path="/"
                        element={
                            <div>
                                <Link to="/fighting-stance">
                                    <button>
                                        Start Fighting Stance Practice
                                    </button>
                                </Link>
                            </div>
                        }
                    />
                    <Route
                        path="/fighting-stance"
                        element={<FightingStance />}
                    />
                    <Route
                        path="/"
                        element={
                            <div>
                                <Link to="/punch">
                                    <button>
                                        Start Punching Practice
                                    </button>
                                </Link>
                            </div>
                        }
                    />
                    <Route
                        path="/punch"
                        element={<Punch />}
                    />
                </Routes>
            </div>
        </Router>
    );
}

export default App;
