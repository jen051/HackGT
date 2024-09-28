import React from "react";
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";
import FightingStance from "./FightingStance";

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
                </Routes>
            </div>
        </Router>
    );
}

export default App;
