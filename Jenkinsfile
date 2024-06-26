pipeline {
    agent any
    environment {
        DOCKER_HUB_USERNAME = 'samansaeed2306'
        DOCKER_HUB_PASSWORD = 'samansaeed1234'
    }

    triggers {
        githubPush(branch: 'master')
    }

    stages {
        stage('Build Docker Image') {
            steps {
                script {
                    // Build Docker image
                    docker.build('mlops_a1:latest', '.')
                }
            }
        }

        stage('Push Docker Image to Docker Hub') {
            steps {
                script {
                    // Login to Docker Hub
                    docker.withRegistry('https://index.docker.io/v1/', DOCKER_HUB_USERNAME, DOCKER_HUB_PASSWORD) {
                        // Push Docker image to Docker Hub
                        docker.image('mlops_a1:latest').push('latest')
                    }
                }
            }
        }
    }

    post {
        success {
            // Send email notification on success
            emailext (
                to: 'samansaeed2306@gmail.com',
                subject: 'Jenkins Build Successful',
                body: 'Your Jenkins build was successful. Docker image pushed to Docker Hub.'
            )
        }
        failure {
            // Send email notification on failure
            emailext (
                to: 'samansaeed2306@gmail.com',
                subject: 'Jenkins Build Failed',
                body: 'Your Jenkins build failed. Please check the Jenkins console output for details.'
            )
        }
    }
}
